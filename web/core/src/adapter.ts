import { ReplayData } from './types';

export interface GameAdapter {
    mount(container: HTMLElement, initialData: ReplayData): void;
    render(step: number, replay: ReplayData): void;
    unmount(): void;
}
