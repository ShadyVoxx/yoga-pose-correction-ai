import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import "./AppPages.css";

const API_BASE = "http://127.0.0.1:3001";

const HistoryPage = () => {
  const { token } = useAuth();
  const [sessions, setSessions] = useState([]);
  const [expanded, setExpanded] = useState({});

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/practices/user/history`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = await res.json();
        if (res.ok) setSessions(data.sessions || []);
      } catch {
        setSessions([]);
      }
    })();
  }, [token]);

  return (
    <div className="app-shell container py-4">
      <h1 className="h4 mb-3">Practice History</h1>
      {sessions.length === 0 ? (
        <div className="app-card p-3">No sessions yet. Start your first practice!</div>
      ) : (
        sessions.map((session) => {
          const totalDuration = Math.round(
            (session.practices || []).reduce((acc, p) => acc + Number(p.duration_seconds || 0), 0)
          );
          return (
            <div key={session.id} className="app-card p-3 mb-3">
              <button className="btn btn-link p-0 text-decoration-none" onClick={() => setExpanded((p) => ({ ...p, [session.id]: !p[session.id] }))}>
                {new Date(session.started_at).toLocaleString()} · {(session.practices || []).length} poses · {totalDuration}s
              </button>
              {expanded[session.id] ? (
                <div className="mt-2">
                  {(session.practices || []).map((practice) => (
                    <div key={practice.id} className="small border-top pt-2 mt-2">
                      <strong>{practice.pose_name}</strong> ({Math.round(practice.duration_seconds || 0)}s) - {practice.stopped_by}
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          );
        })
      )}
      <Link to="/app/dashboard" className="btn btn-outline-dark">Back to dashboard</Link>
    </div>
  );
};

export default HistoryPage;
