const mouse = {
    x: 0,
    y: 0,
    mousedown: false,
    shift: false
}

window.onmousemove = (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
}

window.onmousedown = (e) => {
    mouse.mousedown = true;
}

window.onmouseup = (e) => {
    mouse.mousedown = false;
}

window.onkeydown = (e) => {
    if (e.key === "Shift")
        mouse.shift = true;
}

window.onkeyup = (e) => {
    if (e.key === "Shift")
        mouse.shift = false;
}
