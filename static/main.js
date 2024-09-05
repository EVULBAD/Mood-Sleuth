addEventListener("load", (event) => {
    adjustMarginBottom("getHeight", "adjustMargin")
});
addEventListener("resize", (event) => {adjustMarginBottom("getHeight", "adjustMargin")});

// Definitions.
const fileInput = document.getElementById('CSV-upload');
const fileLabel = document.querySelector('label[for="CSV-upload"]');

fileInput.addEventListener('change', function() {
    if (fileInput.files.length > 0) {
        fileLabel.classList.add('inactive');
        fileLabel.classList.remove('active');
        fileLabel.textContent = 'Uploaded!';
    } else {
        fileLabel.textContent = 'Upload CSV';
    }
});

// Function to get height of an element by id.
function getHeight(id) {
    divElement = document.getElementById(id);
    elemRect = divElement.getBoundingClientRect();
    elemHeight = elemRect.height;
    return elemHeight
}

// Function to center the content on the screen.
function adjustMarginBottom(heightVariable, elem) {
    let w = window.innerWidth
    if (w >= 779) {
        elemHeight = getHeight(heightVariable) +  "px"
        document.getElementById(elem).style.marginBottom = elemHeight;
    } else {
        console.log('adjustMarginBottom: Aborting; Screen not big enough.')
    }
}