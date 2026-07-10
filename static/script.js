const form = document.getElementById("regressionForm");
const visualizeBtn = document.getElementById("visualizeBtn");
const graphsContainer = document.getElementById("graphsContainer");
const graphImg = document.getElementById("graph");
const metricsImg = document.getElementById("metricsGraph");
const errorBox = document.getElementById("errorBox");

let regressionImgUrl = "";
let metricsImgUrl = "";

function showError(message) {
    errorBox.textContent = message;
    errorBox.style.display = "block";
}

function hideError() {
    errorBox.textContent = "";
    errorBox.style.display = "none";
}

form.addEventListener("submit", function (event) {

    event.preventDefault();

    const xVal = document.getElementById("xValues").value.trim();
    const yVal = document.getElementById("yValues").value.trim();


    graphsContainer.style.display = "none";
    graphImg.src = "";
    metricsImg.src = "";
    regressionImgUrl = "";
    metricsImgUrl = "";
    hideError();


    const numberListRegex = /^\s*-?\d+(?:\.\d+)?\s*(?:,\s*-?\d+(?:\.\d+)?\s*)*$/;

    if (!xVal) {
        showError("Please enter values for X.");
        return;
    }
    if (!yVal) {
        showError("Please enter values for Y.");
        return;
    }

    if (!numberListRegex.test(xVal)) {
        showError("X Values must be a comma-separated list of numbers (e.g. 1,2,3,4,5). Only digits, signs (-), decimals (.), and commas are allowed.");
        return;
    }

    if (!numberListRegex.test(yVal)) {
        showError("Y Values must be a comma-separated list of numbers (e.g. 2,4,5,4,5). Only digits, signs (-), decimals (.), and commas are allowed.");
        return;
    }

    const xArr = xVal.split(",").map(i => i.trim()).filter(i => i !== "");
    const yArr = yVal.split(",").map(i => i.trim()).filter(i => i !== "");

    if (xArr.length !== yArr.length) {
        showError(`X and Y must have the same number of elements. Currently X has ${xArr.length} elements and Y has ${yArr.length} elements.`);
        return;
    }

    if (xArr.length < 2) {
        showError("Linear regression requires at least 2 data points.");
        return;
    }

    const firstX = parseFloat(xArr[0]);
    const allXSame = xArr.every(i => parseFloat(i) === firstX);
    if (allXSame) {
        showError("Cannot compute regression: all X values are identical (zero variance).");
        return;
    }

    fetch("/predict", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            x: xVal,
            y: yVal
        })

    })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || "An error occurred during calculation.");
                });
            }
            return response.json();
        })
        .then(data => {
            document.getElementById("equation").innerHTML = data.equation;
            document.getElementById("mae").innerHTML = data.mae;
            document.getElementById("mse").innerHTML = data.mse;
            document.getElementById("rmse").innerHTML = data.rmse;
            document.getElementById("r2").innerHTML = data.r2;

            regressionImgUrl = data.regression_image;
            metricsImgUrl = data.metrics_image;

            // Target the src attribute of <img> tags immediately
            graphImg.src = regressionImgUrl;
            metricsImg.src = metricsImgUrl;
        })
        .catch(err => {
            showError(err.message);
        });

});

visualizeBtn.addEventListener("click", function () {
    if (graphImg.src && metricsImg.src) {
        graphsContainer.style.display = "block";
    } else {
        alert("Please calculate regression first before visualizing!");
    }
});