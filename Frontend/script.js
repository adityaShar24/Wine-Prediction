async function predictWine() {
    const requiredFields = [
        'fixed_acidity', 'volatile_acidity', 'citric_acid', 
        'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 
        'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'
    ];

    const formData = {};
    let isValid = true;

    // Validation check for empty fields
    requiredFields.forEach(field => {
        const value = document.getElementById(field).value.trim();
        if (!value) {
            isValid = false;
            document.getElementById('result').innerHTML = `Error: Please fill in all fields.`;
        } else {
            formData[field] = parseFloat(value);
        }
    });

    if (!isValid) return; // Stop execution if any field is empty

    try {
        const response = await fetch('http://127.0.0.1:8000/predict/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const result = await response.json();
        document.getElementById('result').innerHTML = 
            `Prediction: ${result.prediction}`;
    } catch (error) {
        document.getElementById('result').innerHTML = `Error: Failed to fetch prediction.`;
    }
}
