function checkCyber() {
    let text = document.getElementById("cyberText").value;

    fetch("/api/cyber", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text: text})
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("cyberResult").innerHTML =
            "Prediction: " + data.prediction +
            "<br>Confidence: " + (data.confidence*100).toFixed(2) + "%";
    });
}

function checkLegal() {
    let text = document.getElementById("legalText").value;

    fetch("/api/legal", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text: text})
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("legalResult").innerHTML =
            "Prediction: " + data.prediction;
    });
}

function checkHealth() {

    let symptoms = {
        "irregular periods": parseInt(document.getElementById("irregular").value),
        "fatigue": parseInt(document.getElementById("fatigue").value),
        "weight gain": parseInt(document.getElementById("weight").value)
    };

    fetch("/api/health/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({symptoms: symptoms})
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("healthResult").innerHTML =
            "Prediction: " + data.prediction +
            "<br>Risk Score: " + data.risk_score_percentage + "%";
    });
}

function searchCourses() {
    let query = document.getElementById("searchBox").value;

    fetch("/api/search?q=" + query)
    .then(res => res.json())
    .then(data => {
        document.getElementById("courseResults").innerHTML =
            "Found " + data.total_results + " courses";
    });
}