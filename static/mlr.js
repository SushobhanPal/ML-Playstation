document.addEventListener("DOMContentLoaded", () => {
    // Menu Selectors
    const menuToggle = document.getElementById("menuToggle");
    const menuClose = document.getElementById("menuClose");
    const sidebar = document.getElementById("sidebar");
    const overlay = document.getElementById("overlay");

    // Hamburger Menu Handlers
    function toggleMenu() {
        sidebar.classList.toggle("active");
        overlay.classList.toggle("active");
        menuToggle.classList.toggle("active");
    }

    function closeMenu() {
        sidebar.classList.remove("active");
        overlay.classList.remove("active");
        menuToggle.classList.remove("active");
    }

    menuToggle.addEventListener("click", toggleMenu);
    menuClose.addEventListener("click", closeMenu);
    overlay.addEventListener("click", closeMenu);

    // MLR Configuration Selectors
    const form = document.getElementById("mlrForm");
    const addFeatureBtn = document.getElementById("addFeatureBtn");
    const featuresWrapper = document.getElementById("featuresWrapper");
    const coefficientsBody = document.getElementById("coefficientsBody");
    const visualizeBtn = document.getElementById("visualizeBtn");
    const graphsContainer = document.getElementById("graphsContainer");
    const graphImg = document.getElementById("graph");
    const metricsImg = document.getElementById("metricsGraph");
    const errorBox = document.getElementById("errorBox");

    let mlrImgUrl = "";
    let metricsImgUrl = "";
    let featureIndex = 2; // Starts with X1 and X2 loaded

    function showError(message) {
        errorBox.textContent = message;
        errorBox.style.display = "block";
        errorBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function hideError() {
        errorBox.textContent = "";
        errorBox.style.display = "none";
    }

    // Dynamic Feature Adding
    addFeatureBtn.addEventListener("click", () => {
        featureIndex++;
        
        const featureItem = document.createElement("div");
        featureItem.className = "feature-item";
        featureItem.setAttribute("data-index", featureIndex);
        featureItem.id = `featureItem_${featureIndex}`;
        
        featureItem.innerHTML = `
            <div class="feature-label-row">
                <label for="xValue${featureIndex}">Feature X${featureIndex} Values</label>
                <button type="button" class="remove-feature-btn" onclick="removeFeatureElement(${featureIndex})">&times; Remove</button>
            </div>
            <div class="feature-input-row">
                <input type="text" id="xValue${featureIndex}" class="x-feature-input" placeholder="e.g. 1, 2, 3, 4, 5" required>
            </div>
        `;
        
        featuresWrapper.appendChild(featureItem);
    });

    // Make remove function globally accessible since it is referenced in inline onclicks
    window.removeFeatureElement = function(index) {
        const itemToRemove = document.getElementById(`featureItem_${index}`) || document.querySelector(`.feature-item[data-index="${index}"]`);
        if (itemToRemove) {
            itemToRemove.remove();
            reindexFeatures();
        }
    };

    // Reindex features after removal to keep consecutive X1, X2, X3... order
    function reindexFeatures() {
        const items = featuresWrapper.querySelectorAll(".feature-item");
        featureIndex = 0;
        
        items.forEach((item) => {
            featureIndex++;
            item.setAttribute("data-index", featureIndex);
            item.id = `featureItem_${featureIndex}`;
            
            // Update Label
            const label = item.querySelector("label");
            label.setAttribute("for", `xValue${featureIndex}`);
            label.textContent = `Feature X${featureIndex} Values`;
            
            // Update Input id
            const input = item.querySelector("input");
            input.id = `xValue${featureIndex}`;
            
            // Update Remove Button if present (first item won't have one usually, but let's re-bind onclicks)
            const removeBtn = item.querySelector(".remove-feature-btn");
            if (removeBtn) {
                // Ensure X1 doesn't get a remove button, others do
                if (featureIndex === 1) {
                    removeBtn.remove();
                } else {
                    removeBtn.setAttribute("onclick", `removeFeatureElement(${featureIndex})`);
                }
            } else if (featureIndex > 1) {
                // Add remove button if it's not the first feature and lacks one
                const labelRow = item.querySelector(".feature-label-row");
                if (labelRow) {
                    const btn = document.createElement("button");
                    btn.type = "button";
                    btn.className = "remove-feature-btn";
                    btn.setAttribute("onclick", `removeFeatureElement(${featureIndex})`);
                    btn.innerHTML = "&times; Remove";
                    labelRow.appendChild(btn);
                }
            }
        });
    }

    form.addEventListener("submit", (event) => {
        event.preventDefault();

        const yVal = document.getElementById("yValues").value.trim();
        const featureInputs = featuresWrapper.querySelectorAll(".x-feature-input");

        // Reset display states
        graphsContainer.style.display = "none";
        graphImg.src = "";
        metricsImg.src = "";
        mlrImgUrl = "";
        metricsImgUrl = "";
        visualizeBtn.disabled = true;
        hideError();

        // Validations
        const numberListRegex = /^\s*-?\d+(?:\.\d+)?\s*(?:,\s*-?\d+(?:\.\d+)?\s*)*$/;

        if (!yVal) {
            showError("Please enter values for Y.");
            return;
        }

        if (!numberListRegex.test(yVal)) {
            showError("Target Y values must be a comma-separated list of numbers (e.g. 10,12,15,18,20). Only digits, signs (-), decimals (.), and commas are allowed.");
            return;
        }

        const yArr = yVal.split(",").map(i => parseFloat(i.trim())).filter(i => !isNaN(i));
        const xFeaturesData = [];

        // Validate each feature input
        let validationFailed = false;
        featureInputs.forEach((input, idx) => {
            if (validationFailed) return;
            
            const rawVal = input.value.trim();
            if (!rawVal) {
                showError(`Please enter values for Feature X${idx+1}.`);
                validationFailed = true;
                return;
            }

            if (!numberListRegex.test(rawVal)) {
                showError(`Feature X${idx+1} values must be a comma-separated list of numbers (e.g. 1,2,3,4,5). Only digits, signs (-), decimals (.), and commas are allowed.`);
                validationFailed = true;
                return;
            }

            const colArr = rawVal.split(",").map(i => parseFloat(i.trim())).filter(i => !isNaN(i));
            
            if (colArr.length !== yArr.length) {
                showError(`Length mismatch: Feature X${idx+1} has ${colArr.length} values, but target Y has ${yArr.length} values. All inputs must have matching lengths.`);
                validationFailed = true;
                return;
            }

            xFeaturesData.push(colArr);
        });

        if (validationFailed) return;

        if (yArr.length < 3) {
            showError("Multiple Linear Regression requires at least 3 data points.");
            return;
        }

        // Post request to backend API
        fetch("/predict-mlr", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                X: xFeaturesData,
                Y: yArr
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
            document.getElementById("equation").textContent = data.equation;
            document.getElementById("mae").textContent = data.mae;
            document.getElementById("mse").textContent = data.mse;
            document.getElementById("rmse").textContent = data.rmse;
            document.getElementById("r2").textContent = data.r2;

            // Render Coefficients Table
            coefficientsBody.innerHTML = "";
            data.coefficients.forEach(coef => {
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td>${coef.name}</td>
                    <td>${coef.value.toFixed(4)}</td>
                `;
                coefficientsBody.appendChild(row);
            });

            mlrImgUrl = data.mlr_image;
            metricsImgUrl = data.metrics_image;

            graphImg.src = mlrImgUrl;
            metricsImg.src = metricsImgUrl;

            // Enable Visualize Button
            visualizeBtn.disabled = false;
        })
        .catch(err => {
            showError(err.message);
        });
    });

    visualizeBtn.addEventListener("click", () => {
        if (graphImg.src && metricsImg.src) {
            graphsContainer.style.display = "grid";
            graphsContainer.scrollIntoView({ behavior: 'smooth' });
        } else {
            alert("Please calculate regression first before visualizing!");
        }
    });
});
