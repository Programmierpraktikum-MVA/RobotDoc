(function () {
    var burger = document.querySelector('.burger');
    var menu = document.querySelector('#' + burger.dataset.target);
    burger.addEventListener('click', function () {
        burger.classList.toggle('is-active');
        menu.classList.toggle('is-active');
    });
})();

// START: show / hide output if empty

// Get the outputText element
var outputTextElement = document.getElementById("outputText");

// Get the symptoms element
var outputSymptomsElement = document.getElementById("outputSymptoms");

// Get the output element
var outputElement = document.getElementById("output");

// Check if output is empty
if (outputTextElement.innerHTML.trim() === "" && outputSymptomsElement.innerHTML.trim() === "") {
    // Hide the output element
    outputElement.style.display = "none";
} else {
    // Show the output element
    outputElement.style.display = "block";
}

// END: show / hide output if empty


 // Slider for threshold
 const slider = document.getElementById("mySlider");
 const valueDisplay = document.getElementById("sliderValue");
 slider.addEventListener("input", function () {
     const sliderValue = slider.value;
     valueDisplay.textContent = sliderValue + "%";
 });