addEventListener("load", (event) => {
    adjustMarginBottom("getHeight", "adjustMargin")
});
addEventListener("resize", (event) => {adjustMarginBottom("getHeight", "adjustMargin")});

function getHeight(id) {
    divElement = document.getElementById(id);
    elemRect = divElement.getBoundingClientRect();
    elemHeight = elemRect.height;
    return elemHeight
}

function adjustMarginBottom(heightVariable, elem) {
    elemHeight = getHeight(heightVariable) + "px"
    document.getElementById(elem).style.marginBottom = elemHeight;
}