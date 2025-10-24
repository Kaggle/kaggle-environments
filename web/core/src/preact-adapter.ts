import { h, render, FunctionComponent } from 'preact';
import { GameAdapter } from './adapter';
import { ReplayData } from './types';

interface RendererProps {
    replay: ReplayData;
    step: number;
    agents: any[];
}

export class PreactAdapter implements GameAdapter {
    private container: HTMLElement | null = null;
    private renderer: FunctionComponent<RendererProps>;

    constructor(renderer: FunctionComponent<RendererProps>) {
        this.renderer = renderer;
    }

    mount(container: HTMLElement, initialData: ReplayData): void {
        this.container = container;
        this.render(0, initialData, []);
    }

    render(step: number, replay: ReplayData, agents: any[]): void {
        if (!this.container) return;
        render(h(this.renderer, { replay, step, agents }), this.container);
    }

    unmount(): void {
        if (!this.container) return;
        render(null, this.container);
        this.container = null;
    }
}
