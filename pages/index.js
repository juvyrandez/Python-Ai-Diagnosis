import { useState } from 'react';

export default function Home() {
  const [complaint, setComplaint] = useState('');
  const [age, setAge] = useState('');
  const [systolic_bp, setSystolicBp] = useState('');
  const [diastolic_bp, setDiastolicBp] = useState('');
  const [temperature_c, setTemperature] = useState('');
  const [weight_kg, setWeight] = useState('');
  const [heart_rate_bpm, setHR] = useState('');
  const [resp_rate_cpm, setRR] = useState('');
  const [top3, setTop3] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const isFormEmpty = (
    (complaint.trim() === '') &&
    (age.trim() === '') &&
    (systolic_bp.trim() === '') &&
    (diastolic_bp.trim() === '') &&
    (temperature_c.trim() === '') &&
    (weight_kg.trim() === '') &&
    (heart_rate_bpm.trim() === '') &&
    (resp_rate_cpm.trim() === '')
  );

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (isFormEmpty) {
      setError('Please enter a chief complaint and/or vitals before diagnosing.');
      return;
    }
    setLoading(true);
    setError('');
    setTop3([]);
    try {
      const res = await fetch('/api/diagnose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          complaint,
          age: age || undefined,
          systolic_bp: systolic_bp || undefined,
          diastolic_bp: diastolic_bp || undefined,
          temperature_c: temperature_c || undefined,
          weight_kg: weight_kg || undefined,
          heart_rate_bpm: heart_rate_bpm || undefined,
          resp_rate_cpm: resp_rate_cpm || undefined,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || 'Request failed');
      setTop3(data.top3 || []);
    } catch (err) {
      setError(err.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  const logsUrl = process.env.NEXT_PUBLIC_ML_API_URL
    ? `${process.env.NEXT_PUBLIC_ML_API_URL}/logs`
    : null;

  return (
    <div style={{ padding: 20, maxWidth: 800, margin: '0 auto' }}>
      <h1>AI Diagnosis Assistant</h1>
      <form onSubmit={handleSubmit} style={{ display: 'grid', gap: 12 }}>
        <textarea
          placeholder="Describe your symptoms... (you can include vitals like BP 120/80, Temp 37.5C, HR 88)"
          value={complaint}
          onChange={(e) => setComplaint(e.target.value)}
          rows={5}
          style={{ width: '100%' }}
        />

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
          <input placeholder="Age" value={age} onChange={(e) => setAge(e.target.value)} />
          <input placeholder="Systolic BP" value={systolic_bp} onChange={(e) => setSystolicBp(e.target.value)} />
          <input placeholder="Diastolic BP" value={diastolic_bp} onChange={(e) => setDiastolicBp(e.target.value)} />
          <input placeholder="Temp (C)" value={temperature_c} onChange={(e) => setTemperature(e.target.value)} />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
          <input placeholder="Weight (kg)" value={weight_kg} onChange={(e) => setWeight(e.target.value)} />
          <input placeholder="HR/Pulse (bpm)" value={heart_rate_bpm} onChange={(e) => setHR(e.target.value)} />
          <input placeholder="RR (cpm)" value={resp_rate_cpm} onChange={(e) => setRR(e.target.value)} />
        </div>

        <button type="submit" disabled={loading || isFormEmpty} title={isFormEmpty ? 'Enter complaint and/or vitals' : ''}>
          {loading ? 'Diagnosing…' : 'Diagnose'}
        </button>
      </form>

      {error && <p style={{ color: 'red' }}>{error}</p>}

      {top3.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <strong>Top 3 Diagnoses</strong>
          <ol>
            {top3.map((item, idx) => (
              <li key={idx} style={idx === 0 ? { fontWeight: 'bold' } : undefined}>
                {idx === 0 ? 'Most likely: ' : ''}
                {item.diagnosis} — {(item.probability * 100).toFixed(1)}%
              </li>
            ))}
          </ol>
        </div>
      )}

      {logsUrl && (
        <div style={{ marginTop: 16 }}>
          <a href={logsUrl} target="_blank" rel="noreferrer">Download Predictions Log (CSV)</a>
        </div>
      )}
    </div>
  );
}
