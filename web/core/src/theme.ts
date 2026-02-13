import { createTheme } from '@mui/material/styles';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/700.css';

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
  typography: {
    fontFamily: 'Inter, sans-serif',
    h1: {
      fontSize: '36px',
      lineHeight: '44px',
      fontWeight: '700',
    },
    h2: {
      fontSize: '32px',
      lineHeight: '40px',
      fontWeight: '700',
    },
    h3: {
      fontSize: '28px',
      lineHeight: '36px',
      fontWeight: '700',
    },
    h4: {
      fontSize: '24px',
      lineHeight: '32px',
      fontWeight: '700',
    },
    h5: {
      fontSize: '20px',
      lineHeight: '24px',
      fontWeight: '700',
    },
    h6: {
      fontSize: '16px',
      lineHeight: '20px',
      fontWeight: '700',
    },
  },
  palette: {
    mode: 'dark',
    primary: {
      main: '#ffffff',
    },
    secondary: {
      main: '#ffffff',
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
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          color: '#ffffff',
          borderColor: '#ffffff',
          textTransform: 'none',
          borderRadius: '20px',
          fontFamily: 'Inter',
          padding: '0px 16px',
          height: '36px',
          fontSize: '14px',
          lineHeight: '20px',
          fontWeight: 700,
          width: 'fit-content',
        },
        sizeSmall: {
          height: '28px',
          fontWeight: 400,
          borderColor: 'rgb(189, 193, 198)',
        },
      },
    },
    MuiIcon: {
      styleOverrides: {
        root: {
          height: '24px',
        },
      },
    },
    MuiSlider: {
      styleOverrides: {
        root: {
          height: '3px',
          borderRadius: 0,
          marginBottom: 0,
        },
        rail: {
          color: 'rgb(128, 134, 139)',
        },
        track: {
          color: 'rgb(248, 249, 250)',
          opacity: 1,
        },
        mark: {
          display: 'none',
        },
        markLabel: {
          top: '50%',
          transform: 'translate(-50%, -50%)',
          zIndex: 2,
        },
      },
    },
  },
});

// Light theme variant for games that prefer light mode
export const lightTheme = createTheme({
  breakpoints: themeBreakpoints,
  palette: {
    mode: 'light',
    primary: {
      main: '#202124',
    },
    secondary: {
      main: '#202124',
    },
    background: {
      default: '#ffffff',
      paper: '#f5f5f5',
    },
    text: {
      primary: '#202124',
    },
  },
});
