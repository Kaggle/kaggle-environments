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

    test = environment.steps[step][0].observation.your_last_action

    if (test !=null) {
        p1move = environment.steps[step][0].observation.your_last_action;
        p2move = environment.steps[step][0].observation.opponent_last_action;
        p1score = environment.steps[step][0].observation.your_last_score;

        p1score = 0;
        p2score = 0;
        ties = 0;
        for(i = 1; i < environment.steps.length; ++i){
            score = environment.steps[step][0].observation.your_last_score;
            if(score == 0) {
                p2score += 1;
            } else if(score == 1) {
                p1score += 1;
            } else {
                ties += 1;
            }
        }

        ctx = canvas.getContext("2d");

        // Player Row

        ctx.font = "20px sans-serif";
        ctx.fillStyle = "#FFFFFF";
        ctx.fillText("Player 1",0,20)
        ctx.fillText("Player 2",170,20)

        // Emoji Row

        ctx.font = "60px sans-serif";
        ctx.fillStyle = "#FFFFFF";
        ctx.fillText(weapon_emoji[p1move], 0,100);
        ctx.fillText("vs", 90,100);
        ctx.fillText(weapon_emoji[p2move], 180,100);

        // Result Row

        ctx.font = "20px sans-serif";
        ctx.fillStyle = "#FFFFFF";

        if (p1score == 1) {
            ctx.fillText("Win", 15,150);
        }

        if (p1score == 0) {
            ctx.fillText("Win", 195,150);
        }

        if (p1score == 0.5) {
            ctx.fillText("Tie", 105,150);
        }

        // Score Row
        ctx.font = "20px sans-serif";
        ctx.fillStyle = "#FFFFFF";
        ctx.fillText(p1score, 20,200);
        ctx.fillText(ties, 110,200);
        ctx.fillText(p2score, 205,200);
    }
}
