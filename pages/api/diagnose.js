function isAnimalBiteComplaint(text) {
  if (!text) return false;
  const t = String(text).toLowerCase();
  const kws = ['animal bite', 'dog bite', 'cat bite', 'monkey bite', 'rat bite', 'fox bite', 'scratch', 'lick', 'rabies'];
  return kws.some(k => t.includes(k));
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  const {
    complaint,
    age,
    systolic_bp,
    diastolic_bp,
    temperature_c,
    weight_kg,
    heart_rate_bpm,
    resp_rate_cpm,
  } = req.body || {};

  const body = {
    complaint,
    age,
    systolic_bp,
    diastolic_bp,
    temperature_c,
    weight_kg,
    heart_rate_bpm,
    resp_rate_cpm,
  };

  const isAnimal = isAnimalBiteComplaint(complaint);
  const endpoint = (process.env.ML_API_URL || '') + (isAnimal ? '/predict-animal-bite' : '/predict');

  try {
    const flaskRes = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await flaskRes.json();
    if (!flaskRes.ok) {
      res.status(flaskRes.status).json({ error: data?.error || 'ML API failed' });
      return;
    }

    if (isAnimal) {
      // Shape for existing UI: show the category as the top prediction
      const cat = data.category || 'Animal Bite Category 2';
      const prob = typeof data.category_confidence === 'number' ? data.category_confidence : 0.9;
      res.status(200).json({
        top3: [ { diagnosis: cat, probability: prob } ],
        category: cat,
        category_confidence: prob,
        treatment: data.treatment,
        urgency_level: data.urgency_level,
      });
      return;
    }

    // General path: pass through top3
    res.status(200).json({ top3: data.top3 || [] });
  } catch (err) {
    res.status(500).json({ error: 'ML API failed' });
  }
}
