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

    const weapon_order = ["Rock", "Paper", "Scissors", "Spock", "Lizard", "Airplane", "Sun", "Moon", "Camera", "Grass", "Fire", "Film", "Spanner", "Toilet", "School", "Air", "Death", "Planet", "Curse", "Guitar", "Lock", "Bowl", "Pickaxe", "Cup", "Peace", "Beer", "Computer", "Rain", "Castle", "Water", "Snake", "TV", "Blood", "Rainbow", "Porcupine", "UFO", "Eagle", "Alien", "Monkey", "Prayer", "King", "Mountain", "Queen", "Satan", "Wizard", "Dragon", "Mermaid", "Diamond", "Police", "Trophy", "Woman", "Money", "Baby", "Devil", "Man", "Link", "Home", "Video Game", "Train", "Math", "Car", "Robot", "Noise", "Heart", "Bicycle", "Electricity", "Tree", "Lightning", "Potato", "Ghost", "Duck", "Power", "Wolf", "Microscope", "Cat", "Nuke", "Chicken", "Cloud", "Fish", "Truck", "Spider", "Helicopter", "Bee", "Bomb", "Brain", "Tornado", "Community", "Sand", "Zombie", "Pit", "Bank", "Chain", "Vampire", "Gun", "Bath", "Law", "Monument", "Baloon", "Pancake", "Sword", "Book"]
    const weapons = {
        "Air": "💨",
        "Airplane": "✈️",
        "Alien": "👽",
        "Baby": "👶🏽",
        "Baloon": "🎈",
        "Bank": "🏦",
        "Bath": "🛁",
        "Bee": "🐝",
        "Beer": "🍺",
        "Bicycle": "🚲",
        "Blood": "💉",
        "Bomb": "💣",
        "Book": "📖",
        "Bowl": "🥣",
        "Brain": "🧠",
        "Camera": "📷",
        "Car": "🚗",
        "Castle": "🏰",
        "Cat": "🐈",
        "Chain": "⛓️",
        "Chicken": "🐓",
        "Cloud": "☁️",
        "Community": "👥",
        "Computer": "💻",
        "Cup": "☕",
        "Curse": "🥀",
        "Death": "☠",
        "Devil": "👹",
        "Diamond": "💎",
        "Dragon": "🐉",
        "Duck": "🦆",
        "Eagle": "🦅",
        "Electricity": "💡",
        "Film": "🎥",
        "Fire": "🔥",
        "Fish": "🐟",
        "Ghost": "👻",
        "Grass": "🌱",
        "Guitar": "🎸",
        "Gun": "🔫",
        "Heart": "❤️",
        "Helicopter": "🚁",
        "Home": "🏠",
        "King": "🤴🏼",
        "Law": "⚖️",
        "Lightning": "⚡",
        "Link": "🔗",
        "Lizard": "🦎",
        "Lock": "🔒",
        "Man": "👨🏾",
        "Math": "🔢",
        "Mermaid": "🧜🏽‍♀️",
        "Microscope": "🔬",
        "Money": "💰",
        "Monkey": "🐒",
        "Monument": "🏛️",
        "Moon": "🌙",
        "Mountain": "🏔️",
        "Noise": "🔔",
        "Nuke": "☢️",
        "Pancake": "🥞",
        "Paper": "📄",
        "Peace": "🕊️",
        "Pickaxe": "⛏️",
        "Pit": "🕳️",
        "Planet": "🌎",
        "Police": "👮🏽‍♀️",
        "Porcupine": "🦔",
        "Potato": "🥔",
        "Power": "🔋",
        "Prayer": "🙏🏽",
        "Queen": "👸🏽",
        "Rain": "🌧️",
        "Rainbow": "🌈",
        "Robot": "🤖",
        "Rock": "👊",
        "Sand": "🏖️",
        "Satan": "😈",
        "School": "🏫",
        "Scissors": "✂️",
        "Snake": "🐍",
        "Spanner": "🔧",
        "Spider": "🕷️",
        "Spock": "🖖",
        "Sun": "☀️",
        "Sword": "🗡️",
        "TV": "📺",
        "Toilet": "🚽",
        "Tornado": "🌪️",
        "Train": "🚂",
        "Tree": "🌲",
        "Trophy": "🏆",
        "Truck": "🚚",
        "UFO": "🛸",
        "Vampire": "🧛🏽‍♂️",
        "Video Game": "🎮",
        "Water": "💧",
        "Wizard": "🧙🏼‍♂️",
        "Wolf": "🐺",
        "Woman": "👩🏻",
        "Zombie": "🧟‍♂️"
    }

    // Common Dimensions.
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
    canvas.width = canvasSize;
    canvas.height = canvasSize;
    c.clearRect(0, 0, canvas.width, canvas.height);

    // ------------------------------------------------------------------------------------//

    const last_action = environment.steps[step][0].observation.your_last_action
    if (last_action != null) {
        const p1move = environment.steps[step][0].observation.your_last_action;
        const p2move = environment.steps[step][0].observation.opponent_last_action;
        const p1score = environment.steps[step][0].observation.your_last_score;

        const ctx = canvas.getContext("2d");
        const label_x = 0;
        const player1_x = 250;
        const player2_x = 500;
        const middle_x = (player1_x + player2_x) / 2 + 30;
        const label_y = 40;
        const weapon_id_y = 80;
        const weapon_name_y = 120;
        const weapon_icon_y = 160;
        const result_y = 200;
        const score_y = 240;
        const p1_weapon = weapon_order[p1move];
        const p2_weapon = weapon_order[p2move];

        ctx.font = "30px sans-serif";
        ctx.fillStyle = "#FFFFFF";

        // Player Row
        ctx.fillText("Player 1", player1_x, label_y)
        ctx.fillText("Player 2", player2_x, label_y)

        // Weapon id Row

        ctx.fillText("Weapon ID:", label_x, weapon_id_y);
        ctx.fillText(p1move, player1_x, weapon_id_y);
        ctx.fillText(p2move, player2_x, weapon_id_y);

        // Weapon name Row
        ctx.fillText("Weapon name:", label_x, weapon_name_y);
        ctx.fillText(p1_weapon, player1_x, weapon_name_y);
        ctx.fillText("vs", middle_x,weapon_name_y);
        ctx.fillText(p2_weapon, player2_x, weapon_name_y);

        // Emoji Row

        ctx.fillText("Weapon icon:", label_x, weapon_icon_y);
        ctx.fillText(weapons[p1_weapon], player1_x, weapon_icon_y);
        ctx.fillText(weapons[p2_weapon], player2_x, weapon_icon_y);

        // Result Row
        ctx.fillText("Result:", label_x, result_y);
        if (p1score === 1) {
            ctx.fillText("Win", player1_x, result_y);
        } else if (p1score === 0) {
            ctx.fillText("Win", player2_x, result_y);
        } else {
            ctx.fillText("Tie", middle_x, result_y);
        }

        // Score Row
        ctx.fillText("Score:", label_x, score_y);
        ctx.fillText(environment.steps[step][0].observation.your_score, player1_x, score_y);
        ctx.fillText(environment.steps[step][0].observation.opponent_score, player2_x, score_y);
    }
}
