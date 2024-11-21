import type { Config } from 'tailwindcss';

const config = {
	darkMode: ['class'],
	content: ['./pages/**/*.{ts,tsx}', './components/**/*.{ts,tsx}', './app/**/*.{ts,tsx}', './src/**/*.{ts,tsx}'],
	prefix: '',
	theme: {
		container: {
			center: true,
			padding: '2rem',
			screens: {
				'2xl': '1400px'
			}
		},
		extend: {
			fontFamily: {
				commissioner: ['Commissioner'],
				sans: ['var(--font-geist-sans)'],
				mono: ['var(--font-geist-mono)']
			},
			backgroundImage: {
				'dark-zinc-mesh': 'radial-gradient(at 0% 75%, hsla(240,5%,26%,0.5) 0px, transparent 50%),radial-gradient(at 100% 25%, hsla(240,4%,16%,0.7) 0px, transparent 50%)',
				'light-gradient': 'linear-gradient(165deg, rgba(255,255,255,1) 0%, rgba(250,250,250,1) 20%, rgba(245,245,245,1) 40%, rgba(2,126,234,0.2) 60%, rgba(250,250,250,1) 100%)',
				'dark-gradient': 'linear-gradient(345deg, rgba(0,0,0,1) 0%, rgba(0,51,102,0.2) 25%, rgba(0,51,102,0.2) 40%, rgba(39,39,42,0.5) 60%, rgba(24,24,27,0.7) 75%, rgba(0,0,0,1) 100%)',
				'blue-gradient-for-svg': 'linear-gradient(170deg, rgba(255,255,255,1) 0%, rgba(250,250,250,1) 50%, rgba(2,126,234,0.15) 70%, rgba(2,126,234,0.3) 100%)'
			},
			colors: {
				blue: {
					'25': '#F5FAFF',
					'50': '#EBF7FF',
					'100': '#DEF3FF',
					'200': '#BEE4FF',
					'300': '#81CAFF',
					'400': '#5AB8FF',
					'500': '#2193FD',
					'600': '#027EEA',
					'700': '#0167DC',
					'800': '#004CAE',
					'900': '#003592',
					'950': '#083366',
					main: '#027EEA',
					light: '#6B8ED8',
					dark: '#004CAE'
				},
				green: {
					'25': '#F6FEF9',
					'50': '#ECFDF3',
					'100': '#D1FADF',
					'200': '#A6F4C5',
					'300': '#6CE9A6',
					'400': '#32D583',
					'500': '#12B76A',
					'600': '#039855',
					'700': '#027948',
					'800': '#05603A',
					'900': '#054F31'
				},
				gray: {
					'50': '#FFFFFF',
					'100': '#FAFAFA',
					'200': '#E0E0E0',
					'300': '#CCCCCC',
					'400': '#B2B2B2',
					'500': '#A0A0A0',
					'600': '#5F5F5F',
					'700': '#4D4D4D',
					'800': '#333333',
					'900': '#1E1E1E',
					'950': '#121212'
				},
				border: 'hsl(var(--border))',
				input: 'hsl(var(--input))',
				ring: 'hsl(var(--ring))',
				background: 'hsl(var(--background))',
				foreground: 'hsl(var(--foreground))',
				primary: {
					DEFAULT: 'hsl(var(--primary))',
					foreground: 'hsl(var(--primary-foreground))'
				},
				secondary: {
					DEFAULT: 'hsl(var(--secondary))',
					foreground: 'hsl(var(--secondary-foreground))'
				},
				destructive: {
					DEFAULT: 'hsl(var(--destructive))',
					foreground: 'hsl(var(--destructive-foreground))'
				},
				muted: {
					DEFAULT: 'hsl(var(--muted))',
					foreground: 'hsl(var(--muted-foreground))'
				},
				accent: {
					DEFAULT: 'hsl(var(--accent))',
					foreground: 'hsl(var(--accent-foreground))'
				},
				popover: {
					DEFAULT: 'hsl(var(--popover))',
					foreground: 'hsl(var(--popover-foreground))'
				},
				card: {
					DEFAULT: 'hsl(var(--card))',
					foreground: 'hsl(var(--card-foreground))'
				},
				chart: {
					'1': 'hsl(var(--chart-1))',
					'2': 'hsl(var(--chart-2))',
					'3': 'hsl(var(--chart-3))',
					'4': 'hsl(var(--chart-4))',
					'5': 'hsl(var(--chart-5))'
				}
			},
			borderRadius: {
				lg: 'var(--radius)',
				md: 'calc(var(--radius) - 2px)',
				sm: 'calc(var(--radius) - 4px)'
			},
			keyframes: {
				'accordion-down': {
					from: {
						height: '0'
					},
					to: {
						height: 'var(--radix-accordion-content-height)'
					}
				},
				'accordion-up': {
					from: {
						height: 'var(--radix-accordion-content-height)'
					},
					to: {
						height: '0'
					}
				},
				progress: {
					'0%': {
						transform: ' translateX(0) scaleX(0)'
					},
					'40%': {
						transform: 'translateX(0) scaleX(0.4)'
					},
					'100%': {
						transform: 'translateX(100%) scaleX(0.5)'
					}
				},
				blink: {
					'0%': {
						opacity: '0.2'
					},
					'20%': {
						opacity: '1'
					},
					'100%': {
						opacity: ' 0.2'
					}
				}
			},
			transformOrigin: {
				'left-right': '0% 50%'
			},
			animation: {
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out',
				progress: 'progress 1s infinite linear',
				blink: 'blink 1.4s infinite both'
			}
		}
	},
	// eslint-disable-next-line @typescript-eslint/no-require-imports
	plugins: [require("tailwindcss-animate")]
} satisfies Config;

export default config;
