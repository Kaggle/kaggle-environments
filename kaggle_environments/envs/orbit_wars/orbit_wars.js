async function renderer(context) {
    const {
        act,
        agents,
        environment,
        frame,
        height = 400,
        interactive,
        isInteractive,
        parent,
        step,
        update,
        width = 400,
    } = context;

    // Canvas Setup.
    let canvas = parent.querySelector("canvas");
    if (!canvas) {
        canvas = document.createElement("canvas");
        parent.appendChild(canvas);
    }

    let c = canvas.getContext("2d");
    const size = Math.min(width, height);
    canvas.width = size;
    canvas.height = size;
    c.clearRect(0, 0, canvas.width, canvas.height);

    // Scale from 100x100 game space to canvas size
    const scale = size / 100.0;

    function drawCircle(x, y, r, color, fill = false) {
        c.beginPath();
        c.arc(x * scale, y * scale, r * scale, 0, 2 * Math.PI);
        c.fillStyle = color;
        c.strokeStyle = color;
        c.lineWidth = 2;
        if (fill) {
            c.fill();
        } else {
            c.stroke();
        }
    }

    function drawText(text, x, y, color, size_pt = 12) {
        c.font = `${size_pt * scale / 4}px sans-serif`;
        c.fillStyle = color;
        c.textAlign = "center";
        c.textBaseline = "middle";
        c.fillText(text, x * scale, y * scale);
    }

    // Draw Background
    c.fillStyle = "#F0F0F0";
    c.fillRect(0, 0, canvas.width, canvas.height);

    // Draw Black Hole
    drawCircle(50, 50, 10, "#111111", true);
    drawCircle(50, 50, 10, "#ff0000", false); // Event horizon line

    const current_step = environment.steps[step];
    const obs = current_step[0].observation;

    const colors = ["#FF4444", "#4444FF", "#44FF44", "#FFFF44", "#888888"]; // P0, P1, P2, P3, Neutral

    // Draw Planets
    if (obs.planets) {
        obs.planets.forEach(p => {
            const id = p[0];
            const x = p[1];
            const y = p[2];
            const r = p[3];
            const owner = p[4];
            const ships = p[5];
            
            const color = owner === -1 ? colors[4] : colors[owner];
            
            // Draw planet body
            drawCircle(x, y, r, color, false);
            c.fillStyle = color + "22"; // Transparent fill
            c.fill();
            
            // Draw ship count
            drawText(ships.toString(), x, y, "#000000", 16);
            // Draw ID
            drawText(`P${id}`, x, y - r - 3, "#555555", 10);
        });
    }

    // Draw Fleets
    if (obs.fleets) {
        obs.fleets.forEach(f => {
            const owner = f[1];
            const x = f[4];
            const y = f[5];
            const ships = f[6];
            
            const color = colors[owner];
            // Draw fleet as a triangle or small dot
            drawCircle(x, y, 1.5, color, true);
            // Draw ship count above fleet
            drawText(ships.toString(), x, y - 3, color, 8);
        });
    }
    
    // Draw Step Info
    c.fillStyle = "#000000";
    c.font = "16px sans-serif";
    c.textAlign = "left";
    c.textBaseline = "top";
    c.fillText(`Step: ${step}`, 10, 10);
}
