addEventListener("load", (event) => {
    adjustMarginBottom("getHeight", "adjustMargin");
    mobileWrapperMargin();
});
addEventListener("resize", (event) => {
    adjustMarginBottom("getHeight", "adjustMargin");
    mobileWrapperMargin();
});

// Definitions.
const fileInput = document.getElementById('CSV-upload');
const fileLabel = document.querySelector('label[for="CSV-upload"]');

// Function to update custom file upload button.
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

// Function to check whether or not page is width of mobile device.
function mobileWidth() {
    let w = window.innerWidth
    if (w >= 780) {
        return false
    } else {
        return true
    }
}

// Function to center the content on the screen.
function adjustMarginBottom(heightVariable, elem) {
    let isMobile = mobileWidth()
    let elemHeight;
    if (!isMobile) {
        elemHeight = getHeight(heightVariable) +  "px"
        document.getElementById(elem).style.marginBottom = elemHeight;
    } else {
        console.log('adjustMarginBottom: Aborting; Screen not big enough.')
    }
}

// Function to add to margin-bottom of wrapper if screen is mobile.
function mobileWrapperMargin() {
    let isMobile = mobileWidth();
    let marginBottom;
    if (isMobile) {
        marginBottom = "calc(2rem + " + getHeight('footer') +  "px)";
        document.getElementById('wrapper').style.marginBottom = marginBottom;
    } else {
        console.log('mobileWrapperMargin: Aborting; Screen too big.');
    }
}