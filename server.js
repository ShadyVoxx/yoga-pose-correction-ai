import 'dotenv/config';
import express from 'express';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

/* ───────────────────────────── Config ───────────────────────────── */

const PORT = process.env.PORT || 3000;
const GEMINI_KEY = process.env.GEMINI_API_KEY;

if (!GEMINI_KEY) {
  console.error('\n❌  GEMINI_API_KEY is not set.');
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(GEMINI_KEY);
const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

/* ──────────────────────────── Poses ─────────────────────────────── */

const POSES = [
  {
    category: 'Standing', id: 'ST-01', name: 'Tadasana (Mountain Pose)',
    steps: [
      "Stand 2 feet apart.",
      "Inhale, lift your arm to shoulder level in front.",
      "Interlock the fingers, turn the wrist outwards, inhale, raise the arms up above your head.",
      "Raise the heels off the floor and balance on the toes as you raise your arms.",
      "Bring the heels down.",
      "Exhale, release the interlock of the fingers and bring the arms down to standing posture."
    ]
  },
  {
    category: 'Standing', id: 'ST-02', name: 'Vrkasana (Tree Pose)',
    steps: [
      "Stand with feet 2 inches apart.",
      "Focus on a point in front. Exhale, hold and bend the right leg then place the foot on the inner side of the left thigh.",
      "The heel should be touching the perineum region.",
      "Inhale and extend the arms up and join the palms together for Namaskar Mudra.",
      "Exhale bring the arms down. Release the right leg and bring it to initial position."
    ]
  },
  {
    category: 'Standing', id: 'ST-03', name: 'Pada-hastasana (Hand-to-Foot Pose)',
    steps: [
      "Stand straight with feet 2 inches apart.",
      "Inhale slowly and raise the arms up.",
      "Stretch up the body from the waist. Exhale and bend forward until both palms rest on the ground.",
      "Stretch the back, to make it straight as much as possible.",
      "Now inhale, come up slowly to the upright position and stretch the arms straight above the head.",
      "Exhale, slowly return to the starting position in the reverse order."
    ]
  },
  {
    category: 'Standing', id: 'ST-04-I', name: 'Ardha Cakrasana (Half-Wheel Pose)',
    steps: [
      "Stand straight with feet 2 inches apart.",
      "Support the back at the sides of the waist with the fingers. Try to keep the elbows parallel.",
      "Drop the head backwards stretching the neck muscles.",
      "As you inhale, bend backwards from the lumbar region.",
      "Exhale and relax. Stay here for 10-30 seconds with normal breathing.",
      "Inhale and slowly come up."
    ]
  },
  {
    category: 'Standing', id: 'ST-05-I', name: 'Trikonasana (Triangle Pose)',
    steps: [
      "Stand with your feet 3 feet apart.",
      "Inhale slowly raise both the arms sideways upto shoulder level.",
      "Turn the right foot towards right side.",
      "Exhale, slowly bend to the right side and place the right hand fingers just behind the right foot.",
      "The left arm straight in line with the right arm. Turn the left palm forward.",
      "Turn your head and gaze at the tip of the left middle finger.",
      "Inhale, slowly come up."
    ]
  }
];

/* ──────────────────────── System Prompt ─────────────────────────── */

function buildStepVerificationPrompt(poseName, currentStepText) {
  return `You are an expert yoga instructor and certified biomechanics analyst. A student is practicing "${poseName}" in a live video feed.

They are currently attempting this specific step in the sequence:
STEP TO VERIFY: "${currentStepText}"

YOUR TASK:
Look at the provided image frame. Analyze if the user has successfully achieved the physical position described in the "STEP TO VERIFY".

OUTPUT FORMAT:
Return a JSON object with this exact structure (no markdown fences, ONLY raw JSON):
{
  "stepComplete": true or false,
  "feedback": "If stepComplete is false, provide 1-2 punchy lines telling them exactly what to fix to achieve the step. If true, provide a 1 line encouraging remark."
}

RULES:
- Be reasonably generous. If they are in the rough position, mark stepComplete: true so they can move to the next step.
- The feedback must describe what you ACTUALLY SEE in the image relative to the required step.
- Keep the feedback extremely concise, maximum 2 sentences.
- Return ONLY the JSON object, no surrounding text.`;
}

/* ────────────────────────── Express App ─────────────────────────── */

const app = express();
app.use(express.json({ limit: '10mb' }));
app.use(express.static(join(__dirname, 'public')));

app.get('/api/poses', (_req, res) => {
  res.json(POSES);
});

app.post('/api/analyze-frame', async (req, res) => {
  try {
    const { poseName, currentStepIndex, imageBase64 } = req.body;

    if (!poseName || currentStepIndex === undefined || !imageBase64) {
      return res.status(400).json({ error: 'Missing required fields: poseName, currentStepIndex, imageBase64' });
    }

    const pose = POSES.find(p => p.name === poseName);
    if (!pose) {
      return res.status(400).json({ error: 'Unknown pose' });
    }

    const stepText = pose.steps[currentStepIndex];
    if (!stepText) {
      return res.status(400).json({ error: 'Invalid step index' });
    }

    const systemPrompt = buildStepVerificationPrompt(poseName, stepText);

    const promptData = [
      { text: systemPrompt },
      {
        inlineData: {
          mimeType: 'image/jpeg',
          data: imageBase64.replace(/^data:image\/\w+;base64,/, ''),
        },
      },
    ];

    let result;
    let retries = 3;
    for (let i = 0; i < retries; i++) {
      try {
        result = await model.generateContent(promptData);
        break;
      } catch (error) {
        const isRateLimit = error.message?.includes('503') || error.message?.includes('quota') || error.message?.includes('429');
        if (isRateLimit && i < retries - 1) {
          const delay = Math.pow(2, i) * 1500;
          await new Promise(resolve => setTimeout(resolve, delay));
        } else {
          throw error;
        }
      }
    }

    const responseText = result.response.text();

    let analysis;
    try {
      analysis = JSON.parse(responseText);
    } catch {
      const jsonMatch = responseText.match(/```(?:json)?\s*([\s\S]*?)```/);
      if (jsonMatch) {
        analysis = JSON.parse(jsonMatch[1].trim());
      } else {
        const braceMatch = responseText.match(/\{[\s\S]*\}/);
        if (braceMatch) {
          analysis = JSON.parse(braceMatch[0]);
        } else {
          throw new Error('Could not parse JSON');
        }
      }
    }

    res.json({ success: true, analysis, stepText });
  } catch (error) {
    console.error('❌ Analysis error:', error.message);
    res.status(500).json({ error: error.message || 'An error occurred during analysis.' });
  }
});

app.listen(PORT, () => {
  console.log(`\n🧘 Yoga Pose Coach (Real-Time Step Verification)`);
  console.log(`   http://localhost:${PORT}\n`);
});
