import { GameAdapter } from './adapter';
import { ReplayData } from './types';
import cssString from './style.css?raw';
import { processEpisodeData } from './transformers';

// Inject CSS for a library bundle/build
(() => {
    if (typeof document === 'undefined') return; // Guard for non-browser environments
    const style = document.createElement('style');
    style.textContent = cssString;
    document.head.appendChild(style);
})();

export class Player {
    private container: HTMLElement;
    private adapter: GameAdapter;
    private replay: ReplayData | null = null;
    private agents: any[] = [];
    private step = 0;
    private playing = false;
    private speed = 500; // ms per step
    private mounted = false;
    private showControls = true;

    // --- Element references ---
    private viewer: HTMLElement;
    private controls: HTMLElement;
    private playPauseButton: HTMLButtonElement;
    private playPauseIconPath: SVGPathElement;
    private prevButton: HTMLButtonElement;
    private nextButton: HTMLButtonElement;
    private stepSlider: HTMLInputElement;
    private stepCounter: HTMLSpanElement;

    constructor(container: HTMLElement, adapter: GameAdapter) {
        this.container = container;
        this.adapter = adapter;

        const playerDiv = document.createElement('div');
        playerDiv.className = 'player';

        this.viewer = document.createElement('div');
        this.viewer.className = 'viewer';

        this.controls = document.createElement('div');
        this.controls.className = 'controls';

        playerDiv.appendChild(this.viewer);
        playerDiv.appendChild(this.controls);
        this.container.innerHTML = '';
        this.container.appendChild(playerDiv);

        this.playPauseButton = this.createButton('play-pause', this.getIconHTML('play'));
        this.playPauseIconPath = this.playPauseButton.querySelector('path')!;

        this.prevButton = this.createButton('prev', this.getIconHTML('prev'));
        this.nextButton = this.createButton('next', this.getIconHTML('next'));

        this.stepSlider = document.createElement('input');
        this.stepSlider.type = 'range';
        this.stepSlider.min = '0';
        this.stepSlider.value = '0';

        this.stepCounter = document.createElement('span');
        this.stepCounter.className = 'step-counter';

        this.controls.appendChild(this.playPauseButton);
        this.controls.appendChild(this.prevButton);
        this.controls.appendChild(this.stepSlider);
        this.controls.appendChild(this.nextButton);
        this.controls.appendChild(this.stepCounter);

        // Wire up event listeners ONCE
        this.playPauseButton.addEventListener('click', () => (this.playing ? this.pause() : this.play()));
        this.prevButton.addEventListener('click', () => this.setStep(this.step - 1));
        this.nextButton.addEventListener('click', () => this.setStep(this.step + 1));
        this.stepSlider.addEventListener('input', (e) => {
            this.pause();
            this.setStep(parseInt((e.target as HTMLInputElement).value, 10));
        });

        this.loadData();
    }

    // Helper to create buttons with SVG icons
    private createButton(id: string, svgPathHTML: string): HTMLButtonElement {
        const button = document.createElement('button');
        button.id = id;
        button.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" viewBox="0 0 24 24" fill="#FFFFFF">
                ${svgPathHTML}
            </svg>
        `;
        return button;
    }

    // Helper to get SVG path data
    private getIconHTML(icon: 'play' | 'pause' | 'prev' | 'next'): string {
        switch (icon) {
            case 'play':
                return `<path d="M8 5v14l11-7z" /><path d="M0 0h24v24H0z" fill="none" />`;
            case 'pause':
                return `<path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" /><path d="M0 0h24v24H0z" fill="none" />`;
            case 'prev':
                return `<path d="M6 18V6h2v12H6zm3.5-6L18 6v12l-8.5-6z" />`;
            case 'next':
                return `<path d="M7 18l8.5-6L7 6v12zM15 6v12h2V6h-2z" />`;
        }
    }

    private loadData() {
        const handleMessage = (event: MessageEvent) => {
            if (!event.data) return;

            if (typeof event.data.controls === 'boolean') {
                this.showControls = event.data.controls;
                this.renderControls();
            }

            let needsRender = false;

            // Update agents if provided
            if (event.data.agents) {
                this.agents = event.data.agents;
                needsRender = true;
            }

            // Update replay object from 'environment'
            if (event.data.environment) {
                if (!this.replay) {
                    this.replay = { name: 'unknown', version: 'unknown', steps: [], configuration: {}, info: {} };
                }
                // Use Object.assign to merge new data without overwriting the whole object
                const { steps, ...rest } = event.data.environment;
                Object.assign(this.replay, rest);
                if (Array.isArray(steps)) {
                    this.replay.steps = steps;
                }
                needsRender = true;
            }

            // Update steps from 'setSteps'
            if (event.data.setSteps && this.replay) {
                this.replay.steps = event.data.setSteps;
                needsRender = true;
            }

            // Overwrite replay object if a full 'replay' is provided
            if (event.data.replay) {
                this.replay = event.data.replay;
                needsRender = true;
            }

            // Update the current step
            if (typeof event.data.step === 'number') {
                this.step = event.data.step;
                needsRender = true;
            }

            // If any data was updated and we have a replay object, call setData.
            if (needsRender && this.replay) {
                this.setData(this.replay, this.agents);
            }
        };
        window.addEventListener('message', handleMessage);

        if (import.meta.env.DEV) {
            const replayFile = import.meta.env.VITE_REPLAY_FILE;
            if (replayFile) {
                fetch(replayFile)
                    .then((res) => res.json())
                    .then((data) => {
                      this.setData(data)

                    })
                    .catch((err) => console.error(`Error fetching ${replayFile}:`, err));
            } else {
                this.viewer.innerHTML = '<div>Waiting for replay data...</div>';
            }
        } else {
            this.viewer.innerHTML = '<div>Loading...</div>';
        }
    }

    private setData(replay: ReplayData, agents: any[] = []) {
        this.replay = replay;
        this.agents = agents;

        if (!this.mounted) {
            this.adapter.mount(this.viewer, this.replay);
            this.mounted = true;
        }


        // TODO(michaelaaron) - clean this up into something sane - we should ideally have this in a single location
        // At the moment this is just a terrible way to tell if processEpisodeData has already been called for steps
        if(this?.replay?.steps && !(this?.replay?.steps as any)?.[0]?.stepType) {
          this.replay.steps = processEpisodeData(this.replay, 'repeated_poker')
        }

        // Always update controls and render the current state.
        this.stepSlider.max = (this.replay.steps.length > 0 ? this.replay.steps.length - 1 : 0).toString();
        this.renderControls();
        this.adapter.render(this.step, this.replay, this.agents);

        this.tick();
    }

    private setStep(step: number) {
        if (!this.replay) return;
        this.step = Math.max(0, Math.min(this.replay.steps.length - 1, step));
        this.adapter.render(this.step, this.replay, this.agents);
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
        if (!this.replay || !this.showControls) {
            this.controls.style.display = 'none';
            return;
        }

        this.controls.style.display = 'flex';
        const maxSteps = this.replay.steps.length - 1;

        // Update only what's necessary
        const newIconHTML = this.getIconHTML(this.playing ? 'pause' : 'play');
        if (this.playPauseIconPath.outerHTML !== newIconHTML) {
            this.playPauseIconPath.outerHTML = newIconHTML;
            this.playPauseIconPath = this.playPauseButton.querySelector('path')!;
        }

        this.prevButton.disabled = this.step === 0;
        this.nextButton.disabled = this.step === maxSteps;

        this.stepSlider.value = this.step.toString();
        this.stepCounter.textContent = `${this.step + 1} / ${maxSteps + 1}`;
    }
}
