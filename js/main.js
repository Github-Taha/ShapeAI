const canvas = document.createElement("canvas");
canvas.classList.add("canvas");
document.body.appendChild(canvas);
const ctx = canvas.getContext("2d");
canvas.width = parseFloat(getComputedStyle(canvas).getPropertyValue("width"));
canvas.height = parseFloat(getComputedStyle(canvas).getPropertyValue("height"));

const triangleValEl = document.querySelector("p.perc.triangle");
const squareValEl = document.querySelector("p.perc.square");
const circleValEl = document.querySelector("p.perc.circle");

const TRIANGLE = [1, 0, 0];
const SQUARE = [0, 1, 0];
const CIRCLE = [0, 0, 1];

let nn = new NeuralNetwork2v2(100, [50], 3);

const gridWidth = 10;
const gridHeight = 10;
const blockWidth = canvas.width / gridWidth;
const blockHeight = canvas.height / gridHeight;
let screen = new Int8Array(gridWidth * gridHeight).fill(0);

function animate () {
    requestAnimationFrame(animate);

    // Update Screen
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let y = 0; y < gridHeight; y++) {
        for (let x = 0; x < gridWidth; x++) {
            if (screen[y * gridWidth + x])
                ctx.fillStyle = "white";
            else
                ctx.fillStyle = "black";

            ctx.fillRect(x * blockWidth, y * blockHeight, blockWidth, blockHeight);
        }
    }

    // Calculate Mouse
    const mouseX = mouse.x - canvas.offsetLeft + canvas.width / 2;
    const mouseY = mouse.y - canvas.offsetTop + canvas.height / 2;
    if (mouseX < 0 || mouseY < 0 || mouseX > canvas.width || mouseY > canvas.height)
        return;
    if (!mouse.mousedown)
        return;
    
    const gridX = Math.floor(mouseX / blockWidth);
    const gridY = Math.floor(mouseY / blockHeight);

    screen[gridY * gridWidth + gridX] = mouse.shift ? 0 : 1;

    // Recalculate AI
    const nnVal = nn.feedForward(screen);
    triangleValEl.innerHTML = "Triangle: " + precision(nnVal[0] * 100, 2) + "%";
    triangleValEl.style.setProperty("--width--", nnVal[0] * 100 + "%");
    squareValEl.innerHTML = "Square: &nbsp;&nbsp;" + precision(nnVal[1] * 100, 2) + "%";
    squareValEl.style.setProperty("--width--", nnVal[1] * 100 + "%");
    circleValEl.innerHTML = "Circle: &nbsp;&nbsp;" + precision(nnVal[2] * 100, 2) + "%";
    circleValEl.style.setProperty("--width--", nnVal[2] * 100 + "%");
}

// document.querySelector(".store").onclick = () => {
//     let shape = [...document.querySelectorAll(".shape-select-radio")].filter((v) => v.checked);
//     if (shape.length === 0) {
//         console.log("No shape selected.");
//         return;
//     }
//     shape = shape[0].value;
//     console.log(shape);

//     let output = null;

//     switch (shape) {
//         case "Triangle":
//             output = TRIANGLE;
//             break;
//         case "Square":
//             output = SQUARE;
//             break;
//         case "Circle":
//             output = CIRCLE;
//             break;
//     }

//     trainingData.inputs.push(screen);
//     trainingData.outputs.push(output);
// }

document.querySelector(".clear").onclick = () => {
    screen = new Int8Array(gridWidth * gridHeight).fill(0);

    const nnVal = nn.feedForward(screen);
    triangleValEl.innerHTML = "Triangle: " + precision(nnVal[0] * 100, 2) + "%";
    triangleValEl.style.setProperty("--width--", nnVal[0] * 100 + "%");
    squareValEl.innerHTML = "Square: &nbsp;&nbsp;" + precision(nnVal[1] * 100, 2) + "%";
    squareValEl.style.setProperty("--width--", nnVal[1] * 100 + "%");
    circleValEl.innerHTML = "Circle: &nbsp;&nbsp;" + precision(nnVal[2] * 100, 2) + "%";
    circleValEl.style.setProperty("--width--", nnVal[2] * 100 + "%");
}

document.querySelector(".train").onclick = () => {
    for (let i = 0; i < 10000; i++) {
        let id = Math.floor(Math.random() * trainingData.inputs.length);
        if (id >= trainingData.inputs.length)
            console.log("Invalid Training Example Error: " + id);

        nn.backPropagation(trainingData.inputs[id], trainingData.outputs[id]);
    }

    alert("Done Training");

    const nnVal = nn.feedForward(screen);
    triangleValEl.innerHTML = "Triangle: " + precision(nnVal[0] * 100, 2) + "%";
    triangleValEl.style.setProperty("--width--", nnVal[0] * 100 + "%");
    squareValEl.innerHTML = "Square: &nbsp;&nbsp;" + precision(nnVal[1] * 100, 2) + "%";
    squareValEl.style.setProperty("--width--", nnVal[1] * 100 + "%");
    circleValEl.innerHTML = "Circle: &nbsp;&nbsp;" + precision(nnVal[2] * 100, 2) + "%";
    circleValEl.style.setProperty("--width--", nnVal[2] * 100 + "%");
}

document.querySelector(".run").onclick = () => {
    checkShapeValues(nn);
}

function checkOrValues (nn) {
    for (let i = 0; i < trainingData.inputs.length; i++) {
        let arr = nn.feedForward(trainingData.inputs[i]);
        console.log(`Output Value: ${trainingData.inputs[i][0]} | ${trainingData.inputs[i][1]} = ${arr.indexOf(Math.max(...arr))}`);
    }
}

function checkNotValues (nn) {
    for (let i = 0; i < trainingData.inputs.length; i++) {
        let arr = nn.feedForward(trainingData.inputs[i]);
        console.log(`Output Value: !${trainingData.inputs[i][0]} = ${arr.indexOf(Math.max(...arr))}`);
    }
}

function checkShapeValues (nn) {
    const arr = nn.feedForward(screen);
    const sel = arr.indexOf(Math.max(...arr));
    
    let outputString = "";
    switch (sel) {
        case 0:
            outputString += "Triangle";
            break;
        case 1:
            outputString += "Square";
            break;
        case 2:
            outputString += "Circle";
            break;
    }
    outputString += ": " + precision(Math.max(...arr) * 100, 2) + "%";
    console.log(outputString);
}

function precision (v, p = 2) {
    return Math.round(v * 10 ** p) / 10 ** p;
}

// let str = "";
// for (let a = 0; a < longStr.outputs.length; a++) {
//     str += "[";
//     const len = 3;
//     for (let i = 0; i < len; i++) {
//         str += longStr.outputs[a]["" + i];
//         if (i != len - 1)
//             str += ", ";
//     }
//     str += "],\n";
// }
// console.log(str);

{
    const nnVal = nn.feedForward(screen);
    triangleValEl.innerHTML = "Triangle: " + precision(nnVal[0] * 100, 2) + "%";
    triangleValEl.style.setProperty("--width--", nnVal[0] * 100 + "%");
    squareValEl.innerHTML = "Square: &nbsp;&nbsp;" + precision(nnVal[1] * 100, 2) + "%";
    squareValEl.style.setProperty("--width--", nnVal[1] * 100 + "%");
    circleValEl.innerHTML = "Circle: &nbsp;&nbsp;" + precision(nnVal[2] * 100, 2) + "%";
    circleValEl.style.setProperty("--width--", nnVal[2] * 100 + "%");
}

requestAnimationFrame(animate);
