// frontend/src/App.jsx
import React, { useState, useEffect } from "react";

function App() {
  const [prediction, setPrediction] = useState(null);
  const [recent, setRecent] = useState([]);
  const [loading, setLoading] = useState(false);
  const [useModel, setUseModel] = useState(false);

  async function fetchPredict() {
    setLoading(true);
    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({source:"storage", take:8, use_model: useModel})
      });
      const data = await resp.json();
      setPrediction(data.prediction);
      setRecent(data.recent);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(()=> {
    fetchPredict();
    const iv = setInterval(fetchPredict, 5000); // poll every 5s (adjust)
    return ()=>clearInterval(iv);
  }, [useModel]);

  return (
    <div style={{padding:20, fontFamily:"Arial"}}>
      <h2>Wingo VIP — Live Prediction</h2>
      <div>
        <label>
          <input type="checkbox" checked={useModel} onChange={e=>setUseModel(e.target.checked)} />
          Use trained model (if available)
        </label>
        <button onClick={fetchPredict} disabled={loading} style={{marginLeft:10}}>
          Refresh
        </button>
      </div>
      <div style={{marginTop:20}}>
        {prediction ? (
          <div style={{border:"1px solid #ddd", padding:12, borderRadius:8, display:"inline-block"}}>
            <div><strong>Method:</strong> {prediction.method}</div>
            <div style={{marginTop:8}}>
              <strong>Prediction →</strong> Size: {prediction.size} — Color: {prediction.color} — Numbers: {prediction.numbers ? prediction.numbers.join(",") : "N/A"}
            </div>
          </div>
        ) : <div>No prediction</div>}
      </div>

      <div style={{marginTop:20}}>
        <h4>Recent</h4>
        <div style={{display:"flex", gap:8}}>
          {recent && recent.length ? recent.map((r,idx)=>(
            <div key={idx} style={{padding:8, border:"1px solid #eee", borderRadius:6}}>
              <div><strong>{r.number}</strong></div>
              <div style={{fontSize:12, color:"#666"}}>{(r.colors||[]).join(", ")}</div>
            </div>
          )) : <div>No recent</div>}
        </div>
      </div>
    </div>
  );
}

export default App;
