import express from 'express';
import fs from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json({ limit: '10mb' }));

const DATA_FILE = join(__dirname, 'training_data.json');

app.post('/api/record-pose', (req, res) => {
    const { poseName, stepIndex, landmarks } = req.body;
    if (!poseName || stepIndex === undefined || !landmarks) {
        return res.status(400).json({ error: 'Missing poseName, stepIndex, or landmarks' });
    }

    // Flatten the landmarks for ML training
    // ML expects a flat array of numbers (33 * 3 = 99 features)
    const features = [];
    landmarks.forEach(lm => {
        features.push(lm.x, lm.y, lm.z);
    });

    const entry = {
        poseName,
        stepIndex,
        label: `${poseName}_Step${stepIndex}`,
        features
    };

    // Read existing data
    let data = [];
    if (fs.existsSync(DATA_FILE)) {
        try {
            data = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
        } catch (e) {
            console.error('Error reading existing JSON, starting fresh.');
        }
    }

    data.push(entry);
    fs.writeFileSync(DATA_FILE, JSON.stringify(data, null, 2), 'utf8');

    console.log(`✅ Recorded ${entry.label} (Total records: ${data.length})`);
    res.json({ success: true, totalRecords: data.length });
});

const PORT = 3001;
app.listen(PORT, () => {
    console.log(`\n🎯 Data Collector Running on http://localhost:${PORT}`);
    console.log(`   Send POST to /api/record-pose with MediaPipe landmarks.`);
});
