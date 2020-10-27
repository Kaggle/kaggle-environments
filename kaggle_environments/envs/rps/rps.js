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

    const sign_names = ["Rock", "Paper", "Scissors", "Spock", "Lizard"]
    const sign_icons = ["\u{1f44a}", "\u{270b}", "\u{2702}\u{fe0f}", "\u{1f596}", "\u{1f98e}"]

    // Common Dimensions.
    const maxWidth = 960;
    const maxHeight = 280;
    const canvasSize = Math.min(height, width);
    const unit = 8;
    const offset = canvasSize > 400 ? canvasSize * 0.1 : unit / 2;
    const cellSize = (canvasSize - offset * 2) / 3;

    // Canvas Setup.
    let canvas = parent.querySelector("canvas");
    if (!canvas) {
        canvas = document.createElement("canvas");
        parent.appendChild(canvas);

        if (interactive) {
            canvas.addEventListener("click", evt => {
                if (!isInteractive()) return;
                const rect = evt.target.getBoundingClientRect();
                const x = evt.clientX - rect.left - offset;
                const y = evt.clientY - rect.top - offset;
                act(Math.floor(x / cellSize) + Math.floor(y / cellSize) * 3);
            });
        }
    }

    canvas.style.cursor = isInteractive() ? "pointer" : "default";

    // Canvas setup and reset.
    let c = canvas.getContext("2d");
    canvas.width = Math.min(maxWidth, width);
    canvas.height = Math.min(maxHeight, height);
    c.clearRect(0, 0, canvas.width, canvas.height);

    // ------------------------------------------------------------------------------------//

    if (step < environment.steps.length - 1) {
        const state = environment.steps[step + 1]
        const last_state = environment.steps[step]
        const delta_reward = state[0].reward - last_state[0].reward

        const p1_move = state[1].observation.lastOpponentAction;
        const p2_move = state[0].observation.lastOpponentAction;

        const info = environment.info;
        const player1_text = info?.TeamNames?.[0] || "Player 1";
        const player2_text = info?.TeamNames?.[1] || "Player 2";

        const ctx = canvas.getContext("2d");
        const padding = 20;
        const row_width = (Math.min(maxWidth, width) - padding * 2) / 3;
        const label_x = padding;
        const player1_x = padding + row_width;
        const player2_x = padding + 2 * row_width;
        const middle_x = padding + row_width * 1.5;
        const label_y = 40;
        const sign_id_y = 80;
        const sign_name_y = 120;
        const sign_icon_y = 160;
        const result_y = 200;
        const score_y = 240;

        ctx.font = "30px sans-serif";
        ctx.fillStyle = "#FFFFFF";

        // Player Row
        ctx.fillText(player1_text, player1_x, label_y)
        ctx.fillText(player2_text, player2_x, label_y)

        // Action Id Row
        ctx.fillText("Action:", label_x, sign_id_y);
        ctx.fillText(p1_move, player1_x, sign_id_y);
        ctx.fillText(p2_move, player2_x, sign_id_y);

        // Action Name Row
        ctx.fillText("Name:", label_x, sign_name_y);
        ctx.fillText(sign_names[p1_move], player1_x, sign_name_y);
        ctx.fillText(sign_names[p2_move], player2_x, sign_name_y);

        // Emoji Row
        ctx.fillText("Icon:", label_x, sign_icon_y);
        ctx.fillText(sign_icons[p1_move], player1_x, sign_icon_y);
        ctx.fillText(sign_icons[p2_move], player2_x, sign_icon_y);

        // Result Row
        ctx.fillText("Result:", label_x, result_y);
        if (delta_reward === 1) {
            ctx.fillText("Win", player1_x, result_y);
        } else if (delta_reward === -1) {
            ctx.fillText("Win", player2_x, result_y);
        } else {
            ctx.fillText("Tie", middle_x, result_y);
        }

        // Reward Row
        ctx.fillText("Reward:", label_x, score_y);
        ctx.fillText(state[0].reward, player1_x, score_y);
        ctx.fillText(state[1].reward, player2_x, score_y);
    }
}
