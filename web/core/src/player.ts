import { GameAdapter } from './adapter';
import { ReplayData } from './types';
import './style.css';

export class Player {
    private container: HTMLElement;
    private adapter: GameAdapter;
    private replay: ReplayData | null = null;
    private step = 0;
    private playing = false;
    private speed = 500; // ms per step

    private viewer: HTMLElement;
    private controls: HTMLElement;

    constructor(container: HTMLElement, adapter: GameAdapter) {
        this.container = container;
        this.adapter = adapter;

        this.container.innerHTML = `
            <div class="player">
                <div class="viewer"></div>
                <div class="controls"></div>
            </div>
        `;
        this.viewer = this.container.querySelector('.viewer')!;
        this.controls = this.container.querySelector('.controls')!;

        this.loadData();
    }

    private loadData() {
        const handleMessage = (event: MessageEvent) => {
            if (event.data.replay) {
                this.setData(event.data.replay);
            }
        };
        window.addEventListener('message', handleMessage);

        if (import.meta.env.DEV) {
            const replayFile = import.meta.env.VITE_REPLAY_FILE;
            if (replayFile) {
                fetch(replayFile)
                    .then((res) => res.json())
                    .then((data) => this.setData(data))
                    .catch((err) => console.error(`Error fetching ${replayFile}:`, err));
            } else {
                this.viewer.innerHTML = '<div>Loading... (No replay file specified)</div>';
            }
        } else {
             this.viewer.innerHTML = '<div>Loading...</div>';
        }
    }

    private setData(replay: ReplayData) {
        this.replay = replay;
        this.adapter.mount(this.viewer, this.replay);
        this.renderControls();
        this.tick();
    }

    private setStep(step: number) {
        if (!this.replay) return;
        this.step = Math.max(0, Math.min(this.replay.steps.length - 1, step));
        this.adapter.render(this.step, this.replay);
        this.renderControls();
    }

    private play() {
        if (this.playing) return;
        this.playing = true;
        if (this.replay && this.step === this.replay.steps.length - 1) {
            this.setStep(0);
        }
        this.tick();
        this.renderControls();
    }

    private pause() {
        if (!this.playing) return;
        this.playing = false;
        this.renderControls();
    }

    private tick = () => {
        if (!this.playing || !this.replay) return;

        if (this.step >= this.replay.steps.length - 1) {
            this.playing = false;
            this.renderControls();
            return;
        }

        setTimeout(() => {
            this.setStep(this.step + 1);
            this.tick();
        }, this.speed);
    };

    private renderControls() {
        if (!this.replay) {
            this.controls.innerHTML = '';
            return;
        }
        const maxSteps = this.replay.steps.length - 1;

        const playPauseIcon = this.playing
            ? `<path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" /><path d="M0 0h24v24H0z" fill="none" />`
            : `<path d="M8 5v14l11-7z" /><path d="M0 0h24v24H0z" fill="none" />`;

        this.controls.innerHTML = `
            <button id="play-pause">
                <svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" viewBox="0 0 24 24" fill="#FFFFFF">
                    ${playPauseIcon}
                </svg>
            </button>
            <button id="prev" ${this.step === 0 ? 'disabled' : ''}>
                 <svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" viewBox="0 0 24 24" fill="#FFFFFF">
                    <path d="M6 18V6h2v12H6zm3.5-6L18 6v12l-8.5-6z" />
                </svg>
            </button>
            <input type="range" min="0" max="${maxSteps}" value="${this.step}" />
            <button id="next" ${this.step === maxSteps ? 'disabled' : ''}>
                <svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" viewBox="0 0 24 24" fill="#FFFFFF">
                    <path d="M7 18l8.5-6L7 6v12zM15 6v12h2V6h-2z" />
                </svg>
            </button>
            <span class="step-counter">${this.step + 1} / ${maxSteps + 1}</span>
        `;

        this.controls.querySelector('#play-pause')!.addEventListener('click', () => this.playing ? this.pause() : this.play());
        this.controls.querySelector('#prev')!.addEventListener('click', () => this.setStep(this.step - 1));
        this.controls.querySelector('#next')!.addEventListener('click', () => this.setStep(this.step + 1));
        this.controls.querySelector('input[type="range"]')!.addEventListener('input', (e) => {
            this.pause();
            this.setStep(parseInt((e.target as HTMLInputElement).value, 10));
        });
    }
}
