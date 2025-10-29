import { Player, GameAdapter, ReplayData, processEpisodeData } from '@kaggle-environments/core';
import { renderer } from './repeated_poker_renderer.js';
import { render } from 'preact';

class LegacyAdapter implements GameAdapter {
    private container: HTMLElement | null = null;

    mount(container: HTMLElement): void {
        this.container = container;
    }

    render(step: number, replay: ReplayData, agents: any[]): void {
        if (!this.container) return;

        const gameName =
            replay.configuration?.openSpielGameName ??
            replay.configuration?.game ??
            'repeated_poker';
        const processedSteps = processEpisodeData(
            replay.steps,
            replay.info?.stateHistory ?? [],
            gameName
        );
        const processedReplay = {
            ...replay,
            steps: processedSteps as unknown as ReplayData['steps']
        };

        this.container.innerHTML = ''; // Clear container before rendering
        renderer({
            parent: this.container,
            environment: processedReplay,
            step: step,
            agents: agents,
            // These are probably not used by poker but good to have
            width: this.container.clientWidth,
            height: this.container.clientHeight
        });
    }

    unmount(): void {
        if (this.container) {
            render(null, this.container);
        }
        this.container = null;
    }
}

const app = document.getElementById('app');
if (app) {
    new Player(app, new LegacyAdapter());
}
