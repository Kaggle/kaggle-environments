import { createTheme, Theme } from '@mui/material/styles';
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

// Extend MUI Button variants
declare module '@mui/material/Button' {
  interface ButtonPropsVariantOverrides {
    medium: true;
    low: true;
  }
}

// Color constants
const COLORS = {
  KAGGLE_BLACK: '#202124',
  KAGGLE_WHITE: '#FFFFFF',
  GREY_50: '#F8F9FA',
  GREY_100: '#F1F3F4',
  GREY_400: '#BDC1C6',
  GREY_500: '#9AA0A6',
  GREY_600: '#80868B',
  GREY_800: '#3C4043',
  GREY_850: '#2E3033',
  GREY_900: '#202124',
};

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

const themeTypography = {
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
  body1: {
    fontSize: '16px',
    lineHeight: '24px',
    fontWeight: '400',
  },
  body2: {
    fontSize: '14px',
    lineHeight: '20px',
    fontWeight: '400',
  },
};

// Base theme with structural (non-color) styles
const baseTheme = createTheme({
  breakpoints: themeBreakpoints,
  typography: themeTypography,
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: '20px',
          fontFamily: 'Inter',
          padding: '0px 16px 0px 12px',
          height: '36px',
          width: 'fit-content',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '14px',
          lineHeight: '20px',
          fontWeight: '700',
        },
        startIcon: {
          display: 'flex',
          alignItems: 'center',
          marginRight: '8px',
          marginLeft: 0,
        },
        endIcon: {
          display: 'flex',
          alignItems: 'center',
          marginLeft: '8px',
          marginRight: 0,
        },
        sizeSmall: {
          height: '28px',
          fontWeight: 400,
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        },
      },
    },
    MuiIcon: {
      styleOverrides: {
        root: {
          height: '24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        },
      },
    },
    MuiSvgIcon: {
      styleOverrides: {
        root: {
          display: 'block',
        },
      },
    },
    MuiMenuItem: {
      styleOverrides: {
        root: {
          fontSize: '14px',
          height: '36px',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
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
        mark: {
          display: 'none',
        },
        markLabel: {
          top: '50%',
          transform: 'translate(-50%, -50%)',
          zIndex: 2,
        },
        valueLabel: {
          borderRadius: '25px',
          padding: '4px 8px',
          ':before': {
            display: 'none',
          },
        },
      },
    },
  },
});

// Dark theme (default)
export const theme: Theme = createTheme(baseTheme, {
  palette: {
    mode: 'dark',
    primary: {
      main: COLORS.KAGGLE_WHITE,
    },
    secondary: {
      main: COLORS.KAGGLE_WHITE,
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
    text: {
      primary: COLORS.KAGGLE_WHITE,
      secondary: 'rgba(255, 255, 255, 0.7)',
    },
    divider: 'rgba(255, 255, 255, 0.12)',
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          color: COLORS.KAGGLE_WHITE,
        },
      },
      variants: [
        {
          props: { variant: 'medium' },
          style: {
            backgroundColor: 'transparent',
            color: COLORS.KAGGLE_WHITE,
            border: `1px solid ${COLORS.GREY_400}`,
            '&:hover': {
              backgroundColor: 'transparent',
              border: `1px solid ${COLORS.GREY_500}`,
            },
          },
        },
        {
          props: { variant: 'low' },
          style: {
            backgroundColor: 'transparent',
            color: COLORS.KAGGLE_WHITE,
            border: 'none',
            '&:hover': {
              backgroundColor: COLORS.GREY_800,
            },
          },
        },
      ],
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          color: COLORS.KAGGLE_WHITE,
          '&:hover': {
            backgroundColor: COLORS.GREY_800,
          },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        icon: {
          color: COLORS.KAGGLE_WHITE,
        },
      },
    },
    MuiSlider: {
      styleOverrides: {
        rail: {
          color: COLORS.GREY_600,
          opacity: 1,
        },
        track: {
          color: COLORS.GREY_50,
          opacity: 1,
        },
        valueLabel: {
          backgroundColor: COLORS.KAGGLE_WHITE,
          color: COLORS.KAGGLE_BLACK,
        },
        root: {
          margin: 0,
        },
      },
    },
  },
});

// Light theme variant
export const lightTheme: Theme = createTheme(baseTheme, {
  palette: {
    mode: 'light',
    primary: {
      main: COLORS.KAGGLE_BLACK,
    },
    secondary: {
      main: COLORS.KAGGLE_BLACK,
    },
    background: {
      default: COLORS.KAGGLE_WHITE,
      paper: '#f5f5f5',
    },
    text: {
      primary: COLORS.KAGGLE_BLACK,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          color: COLORS.GREY_900,
        },
      },
      variants: [
        {
          props: { variant: 'medium' },
          style: {
            backgroundColor: 'transparent',
            color: COLORS.GREY_900,
            border: `1px solid ${COLORS.GREY_400}`,
            '&:hover': {
              backgroundColor: 'transparent',
              border: `1px solid ${COLORS.GREY_900}`,
            },
          },
        },
        {
          props: { variant: 'low' },
          style: {
            backgroundColor: 'transparent',
            color: COLORS.GREY_900,
            border: 'none',
            '&:hover': {
              backgroundColor: COLORS.GREY_100,
            },
          },
        },
      ],
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          color: COLORS.GREY_900,
          '&:hover': {
            backgroundColor: COLORS.GREY_100,
          },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        icon: {
          color: COLORS.GREY_900,
        },
      },
    },
    MuiSlider: {
      styleOverrides: {
        rail: {
          color: COLORS.GREY_600,
        },
        track: {
          color: COLORS.GREY_900,
          opacity: 1,
        },
        valueLabel: {
          backgroundColor: COLORS.GREY_900,
          color: COLORS.KAGGLE_WHITE,
        },
      },
    },
  },
});
