import { createTheme } from '@mui/material/styles';

// Custom breakpoints matching the Material design guidelines
// See: https://carbon.googleplex.com/kaggle/pages/layout-breakpoints/principles
declare module '@mui/material/styles' {
  interface BreakpointOverrides {
    xs: true;
    sm: true;
    md: true;
    lg: true;
    xl: true;
    // Custom breakpoints
    xs1: true;
    xs2: true;
    xs3: true;
    sm1: true;
    sm2: true;
    sm3: true;
    md1: true;
    md2: true;
    lg1: true;
    lg2: true;
    lg3: true;
    xl1: true;
    // Aliases
    phone: true;
    tablet: true;
    desktop: true;
  }
}

export const themeBreakpoints = {
  values: {
    xs: 0,
    xs1: 360,
    xs2: 400,
    xs3: 480, // phone
    sm: 600,
    sm1: 600,
    sm2: 720,
    sm3: 840, // tablet
    md: 960,
    md1: 960,
    md2: 1024,
    lg: 1280,
    lg1: 1280, // desktop
    lg2: 1440,
    lg3: 1600,
    xl: 1920,
    xl1: 1920,
    // Aliases
    phone: 480,
    tablet: 840,
    desktop: 1280,
  },
};

export const theme = createTheme({
  breakpoints: themeBreakpoints,
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#ce93d8',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
    text: {
      primary: '#ffffff',
      secondary: 'rgba(255, 255, 255, 0.7)',
    },
    divider: 'rgba(255, 255, 255, 0.12)',
  },
});

// Light theme variant for games that prefer light mode
export const lightTheme = createTheme({
  breakpoints: themeBreakpoints,
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#9c27b0',
    },
    background: {
      default: '#ffffff',
      paper: '#f5f5f5',
    },
  },
});
