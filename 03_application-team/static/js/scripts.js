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

// Get the output elements
var outputElement = document.getElementById("output");
var outputResetElement = document.getElementById("outputReset");

// Check if output is empty
if (outputTextElement.innerHTML.trim() === "" && outputSymptomsElement.innerHTML.trim() === "") {
    // Hide the output element
    outputElement.style.display = "none";
    outputResetElement.style.display = "none";
} else {
    // Show the output element
    outputElement.style.display = "block";
    outputResetElement.style.display = "block";
}

// END: show / hide output if empty


 // Slider for threshold
 