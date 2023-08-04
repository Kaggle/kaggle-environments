async function renderer(context) {
    const {
        act,
        agents,
        environment,
        frame,
        height = 800,
        interactive,
        isInteractive,
        parent,
        step,
        update,
        width = 1200,
    } = context;

    // Common Dimensions.
    const maxWidth = 1200;
    const maxHeight = 800;
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

    const state = environment.steps[step]

    const team1_index = state[0].observation.questions.length - 1;
    const team2_index = state[2].observation.questions.length - 1;

    let team1_question = "";
    let team2_question = "";
    if (team1_index >= 0) {
        team1_question = state[0].observation.questions[team1_index];
    }
    if (team2_index >= 0) {
        team2_question = state[2].observation.questions[team2_index];
    }

    let team1_answer = "";
    let team2_answer = "";
    if (state[0].observation.questions.length == state[0].observation.answers.length && team1_index >= 0) {
        team1_answer = state[0].observation.answers[team1_index];
    }
    if (state[2].observation.questions.length == state[2].observation.answers.length && team2_index >= 0) {
        team2_answer = state[2].observation.answers[team2_index];
    }

    let team1_guess = "";
    let team2_guess = "";
    if (state[0].observation.questions.length == state[0].observation.guesses.length && team1_index >= 0) {
        team1_guess= state[0].observation.guesses[team1_index];
    }
    if (state[2].observation.questions.length == state[2].observation.guesses.length && team2_index >= 0) {
        team2_guess = state[2].observation.guesses[team2_index];
    }

    let team1_reward = "";
    let team2_reward = "";
    if (state[0].reward != 0) {
        team1_reward = state[0].reward.toString();
    }
    if (state[2].reward != 0) {
        team2_reward = state[2].reward.toString();
    }

    const info = environment.info;
    const team1_text = info?.TeamNames?.[0] || "Team 1";
    const team2_text = info?.TeamNames?.[1] || "Team 2";

    const ctx = canvas.getContext("2d");
    const padding = 20;
    const row_width = (Math.min(maxWidth, width) - padding * 3 - 100) / 2;
    const label_x = padding;
    const team1_x = padding + 100;
    const team2_x = padding * 2 + row_width + 100;
    const line_height = 40;
    const label_y = 120;
    const question_y = 160;
    const answer_y = 200;
    const guess_y = 240;
    const score_y = 280;

    ctx.font = "20px sans-serif";
    ctx.fillStyle = "#FFFFFF";

    let line = 1;

    // Keyword Row
    ctx.fillText("Keyword: " + state[1].observation.keyword, label_x, line_height * line);

    line += 2;

    // Team Row
    ctx.fillText(team1_text, team1_x, line_height * line);
    ctx.fillText(team2_text, team2_x, line_height *line);

    line++;

    // Question Row
    ctx.fillText("Question:", label_x, question_y);
    let wrappedText1 = wrapText(ctx, team1_question, team1_x, question_y, row_width, line_height);
    wrappedText1.forEach(function(item) {
        ctx.fillText(item[0], item[1], item[2]); 
    })
    let wrappedText2 = wrapText(ctx, team2_question, team2_x, question_y, row_width, line_height);
    wrappedText2.forEach(function(item) {
        ctx.fillText(item[0], item[1], item[2]); 
    })
    /*ctx.fillText(team1_question, team1_x, line_height * line);
    ctx.fillText(team2_question, team2_x, line_height * line);*/

    line += Math.max(wrappedText1.length, wrappedText2.length);
    //line++;

    // Answer Row
    ctx.fillText("Answer:", label_x, line_height * line);
    ctx.fillText(team1_answer, team1_x, line_height * line);
    ctx.fillText(team2_answer, team2_x, line_height * line);

    line++;

    // Guess Row
    ctx.fillText("Guess:", label_x, line_height * line);
    ctx.fillText(team1_guess, team1_x, line_height * line);
    ctx.fillText(team2_guess, team2_x, line_height * line);

    line++;

    // Reward Row
    ctx.fillText("Reward:", label_x, line_height * line);
    ctx.fillText(team1_reward, team1_x, line_height * line);
    ctx.fillText(team2_reward, team2_x, line_height * line);
}

const wrapText = function(ctx, inputText, x, y, maxWidth, lineHeight) {
    let words = inputText.split(" ");
    let line = "";
    let testLine = "";
    let lineArray = [];

    for(var n = 0; n < words.length; n++) {
        testLine += `${words[n]} `;
        let metrics = ctx.measureText(testLine);
        if (metrics.width > maxWidth && n > 0) {
            lineArray.push([line, x, y]);
            y += lineHeight;
            line = `${words[n]} `;
            testLine = `${words[n]} `;
        }
        else {
            line += `${words[n]} `;
        }
        if(n === words.length - 1) {
            lineArray.push([line, x, y]);
        }
    }
    return lineArray;
}
