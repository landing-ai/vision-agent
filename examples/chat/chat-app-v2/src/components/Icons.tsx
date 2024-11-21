/* eslint-disable max-len */
'use client';

import * as React from 'react';

import { cn } from '@/lib/utils';

export function IconLandingAI() {
  return (
    <svg width="24" fill="none" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <rect rx="4" width="24" height="24" fill="white" />
      <path fill="black" d="M5 13.469V17.0762L7.84434 18.6274V15.0202L5 13.469Z" />
      <path fill="black" d="M5 9.2356V12.8428L7.84434 14.3921V10.7868L5 9.2356Z" />
      <path fill="black" d="M5 5.00391V8.60921L7.84434 10.1604V6.55509L5 5.00391Z" />
      <path fill="black" d="M15.1556 15.0202V18.6274L18 17.0762V13.469L15.1556 15.0202Z" />
      <path fill="black" d="M8.38708 10.7868V14.3921L11.2314 12.8428V9.2356L8.38708 10.7868Z" />
      <path fill="black" d="M8.38708 6.55509V10.1604L11.2314 8.60921V5.00391L8.38708 6.55509Z" />
      <path fill="black" d="M10.9421 4.54541L8.11669 3L5.29315 4.54541L8.11669 6.08889L10.9421 4.54541Z" />
      <path fill="black" d="M8.38708 15.3054V18.9127L11.2314 20.4619V16.8566L8.38708 15.3054Z" />
      <path fill="black" d="M11.7742 16.8566V20.4619L14.6186 18.9127V15.3054L11.7742 16.8566Z" />
      <path fill="black" d="M8.67645 14.8506L11.5 16.396L14.3235 14.8506L11.5 13.3071L8.67645 14.8506Z" />
      <path fill="black" d="M17.7144 13.0108L14.889 11.4673L12.0654 13.0108L14.889 14.5542L17.7144 13.0108Z" />
    </svg>
  );
}

export function IconGitHub({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      role="img"
      fill="currentColor"
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <title>GitHub</title>
      <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
    </svg>
  );
}

export function IconGoogle({ className }: React.ComponentProps<'svg'>) {
  return (
    <svg
      id="google"
      width="2443"
      height="2500"
      viewBox="0 0 256 262"
      preserveAspectRatio="xMidYMid"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
    >
      <path
        fill="#4285F4"
        d="M255.878 133.451c0-10.734-.871-18.567-2.756-26.69H130.55v48.448h71.947c-1.45 12.04-9.283 30.172-26.69 42.356l-.244 1.622 38.755 30.023 2.685.268c24.659-22.774 38.875-56.282 38.875-96.027"
      ></path>
      <path
        fill="#34A853"
        d="M130.55 261.1c35.248 0 64.839-11.605 86.453-31.622l-41.196-31.913c-11.024 7.688-25.82 13.055-45.257 13.055-34.523 0-63.824-22.773-74.269-54.25l-1.531.13-40.298 31.187-.527 1.465C35.393 231.798 79.49 261.1 130.55 261.1"
      ></path>
      <path
        fill="#FBBC05"
        d="M56.281 156.37c-2.756-8.123-4.351-16.827-4.351-25.82 0-8.994 1.595-17.697 4.206-25.82l-.073-1.73L15.26 71.312l-1.335.635C5.077 89.644 0 109.517 0 130.55s5.077 40.905 13.925 58.602l42.356-32.782"
      ></path>
      <path
        fill="#EB4335"
        d="M130.55 50.479c24.514 0 41.05 10.589 50.479 19.438l36.844-35.974C195.245 12.91 165.798 0 130.55 0 79.49 0 35.393 29.301 13.925 71.947l42.211 32.783c10.59-31.477 39.891-54.251 74.414-54.251"
      ></path>
    </svg>
  );
}

export function IconSeparator({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="none"
      strokeWidth="1"
      aria-hidden="true"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={cn('size-4', className)}
      shapeRendering="geometricPrecision"
      {...props}
    >
      <path d="M16.88 3.549L7.12 20.451"></path>
    </svg>
  );
}

export function IconLineVertical({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" className={cn('size-4', className)} {...props}>
      <rect width="256" fill="none" height="256" />
      <line
        y1="24"
        x1="128"
        x2="128"
        y2="232"
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function IconArrowUp({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg height="16" viewBox="0 0 16 16" strokeLinejoin="round" className={cn('size-4', className)} {...props}>
      <path
        clipRule="evenodd"
        fillRule="evenodd"
        fill="currentColor"
        d="M8.70711 1.39644C8.31659 1.00592 7.68342 1.00592 7.2929 1.39644L2.21968 6.46966L1.68935 6.99999L2.75001 8.06065L3.28034 7.53032L7.25001 3.56065V14.25V15H8.75001V14.25V3.56065L12.7197 7.53032L13.25 8.06065L14.3107 6.99999L13.7803 6.46966L8.70711 1.39644Z"
      ></path>
    </svg>
  );
}

export function IconArrowDown({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="m205.66 149.66-72 72a8 8 0 0 1-11.32 0l-72-72a8 8 0 0 1 11.32-11.32L120 196.69V40a8 8 0 0 1 16 0v156.69l58.34-58.35a8 8 0 0 1 11.32 11.32Z" />
    </svg>
  );
}

export function IconArrowUpRight({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      strokeLinejoin="round"
      className={cn('size-4', className)}
      {...props}
    >
      <path
        clipRule="evenodd"
        fillRule="evenodd"
        fill="currentColor"
        d="M5.75001 2H5.00001V3.5H5.75001H11.4393L2.21968 12.7197L1.68935 13.25L2.75001 14.3107L3.28034 13.7803L12.4988 4.56182V10.25V11H13.9988V10.25V3C13.9988 2.44772 13.5511 2 12.9988 2H5.75001Z"
      ></path>
    </svg>
  );
}

export function IconArrowRight({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="m221.66 133.66-72 72a8 8 0 0 1-11.32-11.32L196.69 136H40a8 8 0 0 1 0-16h156.69l-58.35-58.34a8 8 0 0 1 11.32-11.32l72 72a8 8 0 0 1 0 11.32Z" />
    </svg>
  );
}

export function IconCheckCircle({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      strokeLinejoin="round"
      className={cn('size-4', className)}
      {...props}
    >
      <path
        clipRule="evenodd"
        fillRule="evenodd"
        fill="currentColor"
        d="M14.5 8C14.5 11.5899 11.5899 14.5 8 14.5C4.41015 14.5 1.5 11.5899 1.5 8C1.5 4.41015 4.41015 1.5 8 1.5C11.5899 1.5 14.5 4.41015 14.5 8ZM16 8C16 12.4183 12.4183 16 8 16C3.58172 16 0 12.4183 0 8C0 3.58172 3.58172 0 8 0C12.4183 0 16 3.58172 16 8ZM11.5303 6.53033L12.0607 6L11 4.93934L10.4697 5.46967L6.5 9.43934L5.53033 8.46967L5 7.93934L3.93934 9L4.46967 9.53033L5.96967 11.0303C6.26256 11.3232 6.73744 11.3232 7.03033 11.0303L11.5303 6.53033Z"
      ></path>
    </svg>
  );
}

export function IconCrossCircle({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      strokeLinejoin="round"
      className={cn('size-4', className)}
      {...props}
    >
      <path
        clipRule="evenodd"
        fillRule="evenodd"
        fill="currentColor"
        d="M14.5 8C14.5 11.5899 11.5899 14.5 8 14.5C4.41015 14.5 1.5 11.5899 1.5 8C1.5 4.41015 4.41015 1.5 8 1.5C11.5899 1.5 14.5 4.41015 14.5 8ZM16 8C16 12.4183 12.4183 16 8 16C3.58172 16 0 12.4183 0 8C0 3.58172 3.58172 0 8 0C12.4183 0 16 3.58172 16 8ZM5.5 11.5607L6.03033 11.0303L8 9.06066L9.96967 11.0303L10.5 11.5607L11.5607 10.5L11.0303 9.96967L9.06066 8L11.0303 6.03033L11.5607 5.5L10.5 4.43934L9.96967 4.96967L8 6.93934L6.03033 4.96967L5.5 4.43934L4.43934 5.5L4.96967 6.03033L6.93934 8L4.96967 9.96967L4.43934 10.5L5.5 11.5607Z"
      ></path>
    </svg>
  );
}

export function IconUser({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M230.92 212c-15.23-26.33-38.7-45.21-66.09-54.16a72 72 0 1 0-73.66 0c-27.39 8.94-50.86 27.82-66.09 54.16a8 8 0 1 0 13.85 8c18.84-32.56 52.14-52 89.07-52s70.23 19.44 89.07 52a8 8 0 1 0 13.85-8ZM72 96a56 56 0 1 1 56 56 56.06 56.06 0 0 1-56-56Z" />
    </svg>
  );
}

export function IconPlus({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M224 128a8 8 0 0 1-8 8h-80v80a8 8 0 0 1-16 0v-80H40a8 8 0 0 1 0-16h80V40a8 8 0 0 1 16 0v80h80a8 8 0 0 1 8 8Z" />
    </svg>
  );
}

export function IconMinus({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" className={cn('size-4', className)} {...props}>
      <rect width="256" fill="none" height="256" />
      <line
        x1="40"
        y1="128"
        x2="216"
        y2="128"
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function IconArrowElbow({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M200 32v144a8 8 0 0 1-8 8H67.31l34.35 34.34a8 8 0 0 1-11.32 11.32l-48-48a8 8 0 0 1 0-11.32l48-48a8 8 0 0 1 11.32 11.32L67.31 168H184V32a8 8 0 0 1 16 0Z" />
    </svg>
  );
}

export function IconSpinner({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4 animate-spin', className)}
      {...props}
    >
      <path d="M232 128a104 104 0 0 1-208 0c0-41 23.81-78.36 60.66-95.27a8 8 0 0 1 6.68 14.54C60.15 61.59 40 93.27 40 128a88 88 0 0 0 176 0c0-34.73-20.15-66.41-51.34-80.73a8 8 0 0 1 6.68-14.54C208.19 49.64 232 87 232 128Z" />
    </svg>
  );
}

export function IconMessage({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M216 48H40a16 16 0 0 0-16 16v160a15.84 15.84 0 0 0 9.25 14.5A16.05 16.05 0 0 0 40 240a15.89 15.89 0 0 0 10.25-3.78.69.69 0 0 0 .13-.11L82.5 208H216a16 16 0 0 0 16-16V64a16 16 0 0 0-16-16ZM40 224Zm176-32H82.5a16 16 0 0 0-10.3 3.75l-.12.11L40 224V64h176Z" />
    </svg>
  );
}

export function IconTrash({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M216 48h-40v-8a24 24 0 0 0-24-24h-48a24 24 0 0 0-24 24v8H40a8 8 0 0 0 0 16h8v144a16 16 0 0 0 16 16h128a16 16 0 0 0 16-16V64h8a8 8 0 0 0 0-16ZM96 40a8 8 0 0 1 8-8h48a8 8 0 0 1 8 8v8H96Zm96 168H64V64h128Zm-80-104v64a8 8 0 0 1-16 0v-64a8 8 0 0 1 16 0Zm48 0v64a8 8 0 0 1-16 0v-64a8 8 0 0 1 16 0Z" />
    </svg>
  );
}

export function IconRefresh({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M197.67 186.37a8 8 0 0 1 0 11.29C196.58 198.73 170.82 224 128 224c-37.39 0-64.53-22.4-80-39.85V208a8 8 0 0 1-16 0v-48a8 8 0 0 1 8-8h48a8 8 0 0 1 0 16H55.44C67.76 183.35 93 208 128 208c36 0 58.14-21.46 58.36-21.68a8 8 0 0 1 11.31.05ZM216 40a8 8 0 0 0-8 8v23.85C192.53 54.4 165.39 32 128 32c-42.82 0-68.58 25.27-69.66 26.34a8 8 0 0 0 11.3 11.34C69.86 69.46 92 48 128 48c35 0 60.24 24.65 72.56 40H168a8 8 0 0 0 0 16h48a8 8 0 0 0 8-8V48a8 8 0 0 0-8-8Z" />
    </svg>
  );
}

export function IconStop({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M128 24a104 104 0 1 0 104 104A104.11 104.11 0 0 0 128 24Zm0 192a88 88 0 1 1 88-88 88.1 88.1 0 0 1-88 88Zm24-120h-48a8 8 0 0 0-8 8v48a8 8 0 0 0 8 8h48a8 8 0 0 0 8-8v-48a8 8 0 0 0-8-8Zm-8 48h-32v-32h32Z" />
    </svg>
  );
}

export function IconSidebar({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M216 40H40a16 16 0 0 0-16 16v144a16 16 0 0 0 16 16h176a16 16 0 0 0 16-16V56a16 16 0 0 0-16-16ZM40 56h40v144H40Zm176 144H96V56h120v144Z" />
    </svg>
  );
}

export function IconMoon({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M233.54 142.23a8 8 0 0 0-8-2 88.08 88.08 0 0 1-109.8-109.8 8 8 0 0 0-10-10 104.84 104.84 0 0 0-52.91 37A104 104 0 0 0 136 224a103.09 103.09 0 0 0 62.52-20.88 104.84 104.84 0 0 0 37-52.91 8 8 0 0 0-1.98-7.98Zm-44.64 48.11A88 88 0 0 1 65.66 67.11a89 89 0 0 1 31.4-26A106 106 0 0 0 96 56a104.11 104.11 0 0 0 104 104 106 106 0 0 0 14.92-1.06 89 89 0 0 1-26.02 31.4Z" />
    </svg>
  );
}

export function IconSun({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M120 40V16a8 8 0 0 1 16 0v24a8 8 0 0 1-16 0Zm72 88a64 64 0 1 1-64-64 64.07 64.07 0 0 1 64 64Zm-16 0a48 48 0 1 0-48 48 48.05 48.05 0 0 0 48-48ZM58.34 69.66a8 8 0 0 0 11.32-11.32l-16-16a8 8 0 0 0-11.32 11.32Zm0 116.68-16 16a8 8 0 0 0 11.32 11.32l16-16a8 8 0 0 0-11.32-11.32ZM192 72a8 8 0 0 0 5.66-2.34l16-16a8 8 0 0 0-11.32-11.32l-16 16A8 8 0 0 0 192 72Zm5.66 114.34a8 8 0 0 0-11.32 11.32l16 16a8 8 0 0 0 11.32-11.32ZM48 128a8 8 0 0 0-8-8H16a8 8 0 0 0 0 16h24a8 8 0 0 0 8-8Zm80 80a8 8 0 0 0-8 8v24a8 8 0 0 0 16 0v-24a8 8 0 0 0-8-8Zm112-88h-24a8 8 0 0 0 0 16h24a8 8 0 0 0 0-16Z" />
    </svg>
  );
}

export function IconCopy({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M216 32H88a8 8 0 0 0-8 8v40H40a8 8 0 0 0-8 8v128a8 8 0 0 0 8 8h128a8 8 0 0 0 8-8v-40h40a8 8 0 0 0 8-8V40a8 8 0 0 0-8-8Zm-56 176H48V96h112Zm48-48h-32V88a8 8 0 0 0-8-8H96V48h112Z" />
    </svg>
  );
}

export function IconDiscord({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      version="1.1"
      width="800px"
      height="800px"
      viewBox="0 -28.5 256 256"
      preserveAspectRatio="xMidYMid"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <g>
        <path
          fill="#5865F2"
          fillRule="nonzero"
          d="M216.856339,16.5966031 C200.285002,8.84328665 182.566144,3.2084988 164.041564,0 C161.766523,4.11318106 159.108624,9.64549908 157.276099,14.0464379 C137.583995,11.0849896 118.072967,11.0849896 98.7430163,14.0464379 C96.9108417,9.64549908 94.1925838,4.11318106 91.8971895,0 C73.3526068,3.2084988 55.6133949,8.86399117 39.0420583,16.6376612 C5.61752293,67.146514 -3.4433191,116.400813 1.08711069,164.955721 C23.2560196,181.510915 44.7403634,191.567697 65.8621325,198.148576 C71.0772151,190.971126 75.7283628,183.341335 79.7352139,175.300261 C72.104019,172.400575 64.7949724,168.822202 57.8887866,164.667963 C59.7209612,163.310589 61.5131304,161.891452 63.2445898,160.431257 C105.36741,180.133187 151.134928,180.133187 192.754523,160.431257 C194.506336,161.891452 196.298154,163.310589 198.110326,164.667963 C191.183787,168.842556 183.854737,172.420929 176.223542,175.320965 C180.230393,183.341335 184.861538,190.991831 190.096624,198.16893 C211.238746,191.588051 232.743023,181.531619 254.911949,164.955721 C260.227747,108.668201 245.831087,59.8662432 216.856339,16.5966031 Z M85.4738752,135.09489 C72.8290281,135.09489 62.4592217,123.290155 62.4592217,108.914901 C62.4592217,94.5396472 72.607595,82.7145587 85.4738752,82.7145587 C98.3405064,82.7145587 108.709962,94.5189427 108.488529,108.914901 C108.508531,123.290155 98.3405064,135.09489 85.4738752,135.09489 Z M170.525237,135.09489 C157.88039,135.09489 147.510584,123.290155 147.510584,108.914901 C147.510584,94.5396472 157.658606,82.7145587 170.525237,82.7145587 C183.391518,82.7145587 193.761324,94.5189427 193.539891,108.914901 C193.539891,123.290155 183.391518,135.09489 170.525237,135.09489 Z"
        ></path>
      </g>
    </svg>
  );
}

export function IconCheck({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="m229.66 77.66-128 128a8 8 0 0 1-11.32 0l-56-56a8 8 0 0 1 11.32-11.32L96 188.69 218.34 66.34a8 8 0 0 1 11.32 11.32Z" />
    </svg>
  );
}

export function IconClockCounterClockwise({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M136,80v43.47l36.12,21.67a8,8,0,0,1-8.24,13.72l-40-24A8,8,0,0,1,120,128V80a8,8,0,0,1,16,0Zm-8-48A95.44,95.44,0,0,0,60.08,60.15C52.81,67.51,46.35,74.59,40,82V64a8,8,0,0,0-16,0v40a8,8,0,0,0,8,8H72a8,8,0,0,0,0-16H49c7.15-8.42,14.27-16.35,22.39-24.57a80,80,0,1,1,1.66,114.75,8,8,0,1,0-11,11.64A96,96,0,1,0,128,32Z"></path>
    </svg>
  );
}

export function IconDownload({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M224 152v56a16 16 0 0 1-16 16H48a16 16 0 0 1-16-16v-56a8 8 0 0 1 16 0v56h160v-56a8 8 0 0 1 16 0Zm-101.66 5.66a8 8 0 0 0 11.32 0l40-40a8 8 0 0 0-11.32-11.32L136 132.69V40a8 8 0 0 0-16 0v92.69l-26.34-26.35a8 8 0 0 0-11.32 11.32Z" />
    </svg>
  );
}

export function IconClose({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M205.66 194.34a8 8 0 0 1-11.32 11.32L128 139.31l-66.34 66.35a8 8 0 0 1-11.32-11.32L116.69 128 50.34 61.66a8 8 0 0 1 11.32-11.32L128 116.69l66.34-66.35a8 8 0 0 1 11.32 11.32L139.31 128Z" />
    </svg>
  );
}

export function IconEdit({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="none"
      strokeWidth={1.5}
      viewBox="0 0 24 24"
      stroke="currentColor"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10"
      />
    </svg>
  );
}

export function IconShare({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="m237.66 106.35-80-80A8 8 0 0 0 144 32v40.35c-25.94 2.22-54.59 14.92-78.16 34.91-28.38 24.08-46.05 55.11-49.76 87.37a12 12 0 0 0 20.68 9.58c11-11.71 50.14-48.74 107.24-52V192a8 8 0 0 0 13.66 5.65l80-80a8 8 0 0 0 0-11.3ZM160 172.69V144a8 8 0 0 0-8-8c-28.08 0-55.43 7.33-81.29 21.8a196.17 196.17 0 0 0-36.57 26.52c5.8-23.84 20.42-46.51 42.05-64.86C99.41 99.77 127.75 88 152 88a8 8 0 0 0 8-8V51.32L220.69 112Z" />
    </svg>
  );
}

export function IconUsers({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M117.25 157.92a60 60 0 1 0-66.5 0 95.83 95.83 0 0 0-47.22 37.71 8 8 0 1 0 13.4 8.74 80 80 0 0 1 134.14 0 8 8 0 0 0 13.4-8.74 95.83 95.83 0 0 0-47.22-37.71ZM40 108a44 44 0 1 1 44 44 44.05 44.05 0 0 1-44-44Zm210.14 98.7a8 8 0 0 1-11.07-2.33A79.83 79.83 0 0 0 172 168a8 8 0 0 1 0-16 44 44 0 1 0-16.34-84.87 8 8 0 1 1-5.94-14.85 60 60 0 0 1 55.53 105.64 95.83 95.83 0 0 1 47.22 37.71 8 8 0 0 1-2.33 11.07Z" />
    </svg>
  );
}

export function IconExternalLink({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M224 104a8 8 0 0 1-16 0V59.32l-66.33 66.34a8 8 0 0 1-11.32-11.32L196.68 48H152a8 8 0 0 1 0-16h64a8 8 0 0 1 8 8Zm-40 24a8 8 0 0 0-8 8v72H48V80h72a8 8 0 0 0 0-16H48a16 16 0 0 0-16 16v128a16 16 0 0 0 16 16h128a16 16 0 0 0 16-16v-72a8 8 0 0 0-8-8Z" />
    </svg>
  );
}

export function IconShareLink({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" className={cn('size-4', className)} {...props}>
      <rect width="256" fill="none" height="256" />
      <circle
        r="32"
        cx="64"
        cy="128"
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle
        r="32"
        cx="176"
        cy="200"
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle
        r="32"
        cy="56"
        cx="176"
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <line
        y1="73.3"
        x2="90.91"
        y2="110.7"
        x1="149.09"
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <line
        x1="90.91"
        y1="145.3"
        y2="182.7"
        x2="149.09"
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function IconChevronUpDown({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M181.66 170.34a8 8 0 0 1 0 11.32l-48 48a8 8 0 0 1-11.32 0l-48-48a8 8 0 0 1 11.32-11.32L128 212.69l42.34-42.35a8 8 0 0 1 11.32 0Zm-96-84.68L128 43.31l42.34 42.35a8 8 0 0 0 11.32-11.32l-48-48a8 8 0 0 0-11.32 0l-48 48a8 8 0 0 0 11.32 11.32Z" />
    </svg>
  );
}

export function IconLoading({ className }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="none"
      aria-hidden="true"
      viewBox="0 0 100 101"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-8 animate-spin fill-gray-600 text-gray-200', className)}
    >
      <path
        fill="currentColor"
        d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z"
      />
      <path
        fill="currentFill"
        d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z"
      />
    </svg>
  );
}

export function IconExclamationTriangle({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      width="15"
      fill="none"
      height="15"
      viewBox="0 0 15 15"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path
        clipRule="evenodd"
        fillRule="evenodd"
        fill="currentColor"
        d="M8.4449 0.608765C8.0183 -0.107015 6.9817 -0.107015 6.55509 0.608766L0.161178 11.3368C-0.275824 12.07 0.252503 13 1.10608 13H13.8939C14.7475 13 15.2758 12.07 14.8388 11.3368L8.4449 0.608765ZM7.4141 1.12073C7.45288 1.05566 7.54712 1.05566 7.5859 1.12073L13.9798 11.8488C14.0196 11.9154 13.9715 12 13.8939 12H1.10608C1.02849 12 0.980454 11.9154 1.02018 11.8488L7.4141 1.12073ZM6.8269 4.48611C6.81221 4.10423 7.11783 3.78663 7.5 3.78663C7.88217 3.78663 8.18778 4.10423 8.1731 4.48612L8.01921 8.48701C8.00848 8.766 7.7792 8.98664 7.5 8.98664C7.2208 8.98664 6.99151 8.766 6.98078 8.48701L6.8269 4.48611ZM8.24989 10.476C8.24989 10.8902 7.9141 11.226 7.49989 11.226C7.08567 11.226 6.74989 10.8902 6.74989 10.476C6.74989 10.0618 7.08567 9.72599 7.49989 9.72599C7.9141 9.72599 8.24989 10.0618 8.24989 10.476Z"
      ></path>
    </svg>
  );
}

export function IconImage({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      strokeLinejoin="round"
      data-testid="geist-icon"
      className={cn('size-4', className)}
      {...props}
    >
      <path
        clipRule="evenodd"
        fillRule="evenodd"
        fill="currentColor"
        d="M14.5 2.5H1.5V9.18933L2.96966 7.71967L3.18933 7.5H3.49999H6.63001H6.93933L6.96966 7.46967L10.4697 3.96967L11.5303 3.96967L14.5 6.93934V2.5ZM8.00066 8.55999L9.53034 10.0897L10.0607 10.62L9.00001 11.6807L8.46968 11.1503L6.31935 9H3.81065L1.53032 11.2803L1.5 11.3106V12.5C1.5 13.0523 1.94772 13.5 2.5 13.5H13.5C14.0523 13.5 14.5 13.0523 14.5 12.5V9.06066L11 5.56066L8.03032 8.53033L8.00066 8.55999ZM4.05312e-06 10.8107V12.5C4.05312e-06 13.8807 1.11929 15 2.5 15H13.5C14.8807 15 16 13.8807 16 12.5V9.56066L16.5607 9L16.0303 8.46967L16 8.43934V2.5V1H14.5H1.5H4.05312e-06V2.5V10.6893L-0.0606689 10.75L4.05312e-06 10.8107Z"
      ></path>
    </svg>
  );
}

export function IconChevronRight({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      strokeLinejoin="round"
      className={cn('size-4', className)}
      {...props}
    >
      <path
        clipRule="evenodd"
        fillRule="evenodd"
        fill="currentColor"
        d="M5.50001 1.93933L6.03034 2.46966L10.8536 7.29288C11.2441 7.68341 11.2441 8.31657 10.8536 8.7071L6.03034 13.5303L5.50001 14.0607L4.43935 13L4.96968 12.4697L9.43935 7.99999L4.96968 3.53032L4.43935 2.99999L5.50001 1.93933Z"
      ></path>
    </svg>
  );
}

export function IconChevronDoubleRight({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      strokeLinejoin="round"
      className={cn('size-4', className)}
      {...props}
    >
      <path
        clipRule="evenodd"
        fillRule="evenodd"
        fill="currentColor"
        d="M12.8536 8.7071C13.2441 8.31657 13.2441 7.68341 12.8536 7.29288L9.03034 3.46966L8.50001 2.93933L7.43935 3.99999L7.96968 4.53032L11.4393 7.99999L7.96968 11.4697L7.43935 12L8.50001 13.0607L9.03034 12.5303L12.8536 8.7071ZM7.85356 8.7071C8.24408 8.31657 8.24408 7.68341 7.85356 7.29288L4.03034 3.46966L3.50001 2.93933L2.43935 3.99999L2.96968 4.53032L6.43935 7.99999L2.96968 11.4697L2.43935 12L3.50001 13.0607L4.03034 12.5303L7.85356 8.7071Z"
      ></path>
    </svg>
  );
}

export function IconTerminalWindow({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      strokeLinejoin="round"
      className={cn('size-4', className)}
      {...props}
    >
      <path
        clipRule="evenodd"
        fillRule="evenodd"
        fill="currentColor"
        d="M1.5 2.5H14.5V12.5C14.5 13.0523 14.0523 13.5 13.5 13.5H2.5C1.94772 13.5 1.5 13.0523 1.5 12.5V2.5ZM0 1H1.5H14.5H16V2.5V12.5C16 13.8807 14.8807 15 13.5 15H2.5C1.11929 15 0 13.8807 0 12.5V2.5V1ZM4 11.1339L4.44194 10.6919L6.51516 8.61872C6.85687 8.27701 6.85687 7.72299 6.51517 7.38128L4.44194 5.30806L4 4.86612L3.11612 5.75L3.55806 6.19194L5.36612 8L3.55806 9.80806L3.11612 10.25L4 11.1339ZM8 9.75494H8.6225H11.75H12.3725V10.9999H11.75H8.6225H8V9.75494Z"
      ></path>
    </svg>
  );
}

export function IconCode({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M69.12,94.15,28.5,128l40.62,33.85a8,8,0,1,1-10.24,12.29l-48-40a8,8,0,0,1,0-12.29l48-40a8,8,0,0,1,10.24,12.3Zm176,27.7-48-40a8,8,0,1,0-10.24,12.3L227.5,128l-40.62,33.85a8,8,0,1,0,10.24,12.29l48-40a8,8,0,0,0,0-12.29ZM162.73,32.48a8,8,0,0,0-10.25,4.79l-64,176a8,8,0,0,0,4.79,10.26A8.14,8.14,0,0,0,96,224a8,8,0,0,0,7.52-5.27l64-176A8,8,0,0,0,162.73,32.48Z"></path>
    </svg>
  );
}

export function IconListUnordered({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      strokeLinejoin="round"
      className={cn('size-4', className)}
      {...props}
    >
      <path
        clipRule="evenodd"
        fillRule="evenodd"
        fill="currentColor"
        d="M2.5 4C3.19036 4 3.75 3.44036 3.75 2.75C3.75 2.05964 3.19036 1.5 2.5 1.5C1.80964 1.5 1.25 2.05964 1.25 2.75C1.25 3.44036 1.80964 4 2.5 4ZM2.5 9.25C3.19036 9.25 3.75 8.69036 3.75 8C3.75 7.30964 3.19036 6.75 2.5 6.75C1.80964 6.75 1.25 7.30964 1.25 8C1.25 8.69036 1.80964 9.25 2.5 9.25ZM3.75 13.25C3.75 13.9404 3.19036 14.5 2.5 14.5C1.80964 14.5 1.25 13.9404 1.25 13.25C1.25 12.5596 1.80964 12 2.5 12C3.19036 12 3.75 12.5596 3.75 13.25ZM6.75 2H6V3.5H6.75H14.25H15V2H14.25H6.75ZM6.75 7.25H6V8.75H6.75H14.25H15V7.25H14.25H6.75ZM6.75 12.5H6V14H6.75H14.25H15V12.5H14.25H6.75Z"
      ></path>
    </svg>
  );
}

export function IconLog({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      strokeLinejoin="round"
      className={cn('size-4', className)}
      {...props}
    >
      <path
        clipRule="evenodd"
        fillRule="evenodd"
        fill="currentColor"
        d="M3 2.5C3 2.22386 3.22386 2 3.5 2H9.08579C9.21839 2 9.34557 2.05268 9.43934 2.14645L11.8536 4.56066C11.9473 4.65443 12 4.78161 12 4.91421V12.5C12 12.7761 11.7761 13 11.5 13H3.5C3.22386 13 3 12.7761 3 12.5V2.5ZM3.5 1C2.67157 1 2 1.67157 2 2.5V12.5C2 13.3284 2.67157 14 3.5 14H11.5C12.3284 14 13 13.3284 13 12.5V4.91421C13 4.51639 12.842 4.13486 12.5607 3.85355L10.1464 1.43934C9.86514 1.15804 9.48361 1 9.08579 1H3.5ZM4.5 4C4.22386 4 4 4.22386 4 4.5C4 4.77614 4.22386 5 4.5 5H7.5C7.77614 5 8 4.77614 8 4.5C8 4.22386 7.77614 4 7.5 4H4.5ZM4.5 7C4.22386 7 4 7.22386 4 7.5C4 7.77614 4.22386 8 4.5 8H10.5C10.7761 8 11 7.77614 11 7.5C11 7.22386 10.7761 7 10.5 7H4.5ZM4.5 10C4.22386 10 4 10.2239 4 10.5C4 10.7761 4.22386 11 4.5 11H10.5C10.7761 11 11 10.7761 11 10.5C11 10.2239 10.7761 10 10.5 10H4.5Z"
      />
    </svg>
  );
}

export function IconOutput({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      strokeLinejoin="round"
      className={cn('size-4', className)}
      {...props}
    >
      <path
        clipRule="evenodd"
        fillRule="evenodd"
        fill="currentColor"
        d="M5 2V1H10V2H5ZM4.75 0C4.33579 0 4 0.335786 4 0.75V1H3.5C2.67157 1 2 1.67157 2 2.5V12.5C2 13.3284 2.67157 14 3.5 14H11.5C12.3284 14 13 13.3284 13 12.5V2.5C13 1.67157 12.3284 1 11.5 1H11V0.75C11 0.335786 10.6642 0 10.25 0H4.75ZM11 2V2.25C11 2.66421 10.6642 3 10.25 3H4.75C4.33579 3 4 2.66421 4 2.25V2H3.5C3.22386 2 3 2.22386 3 2.5V12.5C3 12.7761 3.22386 13 3.5 13H11.5C11.7761 13 12 12.7761 12 12.5V2.5C12 2.22386 11.7761 2 11.5 2H11Z"
      />
    </svg>
  );
}

export function IconGlowingDot({ className, ...props }: React.ComponentProps<'div'>) {
  return <div className={cn('size-3 svg-shadow', className)} {...props} />;
}

export function IconSpark({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" className={cn('size-4', className)} {...props}>
      <path
        fill="currentColor"
        d="M197.58,129.06,146,110l-19-51.62a15.92,15.92,0,0,0-29.88,0L78,110l-51.62,19a15.92,15.92,0,0,0,0,29.88L78,178l19,51.62a15.92,15.92,0,0,0,29.88,0L146,178l51.62-19a15.92,15.92,0,0,0,0-29.88ZM137,164.22a8,8,0,0,0-4.74,4.74L112,223.85,91.78,169A8,8,0,0,0,87,164.22L32.15,144,87,123.78A8,8,0,0,0,91.78,119L112,64.15,132.22,119a8,8,0,0,0,4.74,4.74L191.85,144ZM144,40a8,8,0,0,1,8-8h16V16a8,8,0,0,1,16,0V32h16a8,8,0,0,1,0,16H184V64a8,8,0,0,1-16,0V48H152A8,8,0,0,1,144,40ZM248,88a8,8,0,0,1-8,8h-8v8a8,8,0,0,1-16,0V96h-8a8,8,0,0,1,0-16h8V72a8,8,0,0,1,16,0v8h8A8,8,0,0,1,248,88Z"
      ></path>
    </svg>
  );
}

export function IconThumbsUp({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" className={cn('size-4', className)} {...props}>
      <path
        fill="currentColor"
        d="M234,80.12A24,24,0,0,0,216,72H160V56a40,40,0,0,0-40-40,8,8,0,0,0-7.16,4.42L75.06,96H32a16,16,0,0,0-16,16v88a16,16,0,0,0,16,16H204a24,24,0,0,0,23.82-21l12-96A24,24,0,0,0,234,80.12ZM32,112H72v88H32ZM223.94,97l-12,96a8,8,0,0,1-7.94,7H88V105.89l36.71-73.43A24,24,0,0,1,144,56V80a8,8,0,0,0,8,8h64a8,8,0,0,1,7.94,9Z"
      ></path>
    </svg>
  );
}

export function IconSmileyMeh({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" className={cn('size-4', className)} {...props}>
      <rect width="256" fill="none" height="256" />
      <circle r="96" cx="128" cy="128" fill="none" strokeWidth="16" stroke="currentColor" strokeMiterlimit="10" />
      <line
        x1="88"
        y1="160"
        x2="168"
        y2="160"
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle r="12" cx="92" cy="108" fill="currentColor" />
      <circle r="12" cx="164" cy="108" fill="currentColor" />
    </svg>
  );
}

export function IconThumbsDown({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" className={cn('size-4', className)} {...props}>
      <path
        fill="currentColor"
        d="M239.82,157l-12-96A24,24,0,0,0,204,40H32A16,16,0,0,0,16,56v88a16,16,0,0,0,16,16H75.06l37.78,75.58A8,8,0,0,0,120,240a40,40,0,0,0,40-40V184h56a24,24,0,0,0,23.82-27ZM72,144H32V56H72Zm150,21.29a7.88,7.88,0,0,1-6,2.71H152a8,8,0,0,0-8,8v24a24,24,0,0,1-19.29,23.54L88,150.11V56H204a8,8,0,0,1,7.94,7l12,96A7.87,7.87,0,0,1,222,165.29Z"
      ></path>
    </svg>
  );
}

export function IconCaretDown({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" className={cn('size-4', className)} {...props}>
      <path
        fill="currentColor"
        d="M213.66,101.66l-80,80a8,8,0,0,1-11.32,0l-80-80A8,8,0,0,1,53.66,90.34L128,164.69l74.34-74.35a8,8,0,0,1,11.32,11.32Z"
      ></path>
    </svg>
  );
}

export function IconCaretRight({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" className={cn('size-4', className)} {...props}>
      <path
        fill="currentColor"
        d="M181.66,133.66l-80,80a8,8,0,0,1-11.32-11.32L164.69,128,90.34,53.66a8,8,0,0,1,11.32-11.32l80,80A8,8,0,0,1,181.66,133.66Z"
      ></path>
    </svg>
  );
}

export function IconCaretUp({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" className={cn('size-4', className)} {...props}>
      <path
        fill="currentColor"
        d="M213.66,165.66a8,8,0,0,1-11.32,0L128,91.31,53.66,165.66a8,8,0,0,1-11.32-11.32l80-80a8,8,0,0,1,11.32,0l80,80A8,8,0,0,1,213.66,165.66Z"
      ></path>
    </svg>
  );
}

export function IconCaretDoubleDown({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" className={cn('size-4', className)} {...props}>
      <rect width="256" fill="none" height="256" />
      <polyline
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        points="208 136 128 216 48 136"
      />
      <polyline
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        points="208 56 128 136 48 56"
      />
    </svg>
  );
}

export function IconEmpty({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" className={cn('size-5', className)} {...props}>
      <path
        fill="currentColor"
        d="M198.24,62.63l15.68-17.25a8,8,0,0,0-11.84-10.76L186.4,51.86A95.95,95.95,0,0,0,57.76,193.37L42.08,210.62a8,8,0,1,0,11.84,10.76L69.6,204.14A95.95,95.95,0,0,0,198.24,62.63ZM48,128A80,80,0,0,1,175.6,63.75l-107,117.73A79.63,79.63,0,0,1,48,128Zm80,80a79.55,79.55,0,0,1-47.6-15.75l107-117.73A79.95,79.95,0,0,1,128,208Z"
      ></path>
    </svg>
  );
}

export function IconWarningCircle({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-5', className)}
      {...props}
    >
      <path d="M128,24A104,104,0,1,0,232,128,104.11,104.11,0,0,0,128,24Zm0,192a88,88,0,1,1,88-88A88.1,88.1,0,0,1,128,216Zm-8-80V80a8,8,0,0,1,16,0v56a8,8,0,0,1-16,0Zm20,36a12,12,0,1,1-12-12A12,12,0,0,1,140,172Z"></path>
    </svg>
  );
}

export function IconSearch({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" className={cn('size-4', className)} {...props}>
      <rect fill="none" width="256" height="256" />
      <circle
        r="80"
        cx="112"
        cy="112"
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <line
        x2="224"
        y2="224"
        fill="none"
        x1="168.57"
        y1="168.57"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function IconMultiModel({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      width="16"
      fill="none"
      height="22"
      viewBox="0 0 16 22"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path fill="#81CAFF" d="M9.65935 3.25027L5.08171 0.790039L0.51416 3.25027L5.08171 5.72058L9.65935 3.25027Z" />
      <path fill="#083366" d="M5.50488 6.49048V12.268L10.1229 9.78759V4L5.50488 6.49048Z" />
      <path fill="#027EEA" d="M0 4V9.78759L4.61797 12.268V6.49048L0 4Z" />
      <path fill="#027EEA" d="M0 4V9.78759L4.61797 12.268V6.49048L0 4Z" />
      <path fill="#81CAFF" d="M5.96924 13.0045L10.557 15.4849L15.1447 13.0045L10.557 10.5342L5.96924 13.0045Z" />
      <path fill="#083366" d="M11.0107 16.2201V22.0077L15.6287 19.5172V13.7397L11.0107 16.2201Z" />
      <path fill="#027EEA" d="M5.50488 13.7397V19.5172L10.1229 22.0077V16.2201L5.50488 13.7397Z" />
      <path fill="#083366" d="M0 10.7959V16.5734L4.61797 19.0639V13.2763L0 10.7959Z" />
    </svg>
  );
}

export function IconSingleModel({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      width="11"
      fill="none"
      height="12"
      viewBox="0 0 11 12"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path fill="#81CAFF" d="M0.464355 2.47031L5.05207 4.9507L9.63979 2.47031L5.05207 0L0.464355 2.47031Z" />
      <path fill="#083366" d="M5.50586 5.68596V11.4735L10.1238 8.98307V3.20557L5.50586 5.68596Z" />
      <path fill="#027EEA" d="M0 3.20557V8.98307L4.61797 11.4735V5.68596L0 3.20557Z" />
    </svg>
  );
}

export function IconModelBuilder({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      width="20"
      fill="none"
      height="22"
      viewBox="0 0 20 22"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <g clipPath="url(#clip0_67_62)">
        <path
          strokeWidth="0.8"
          stroke="currentColor"
          strokeOpacity="0.306798"
          d="M0.359924 6.26901V16.8753L9.15517 21.4209V10.833L0.359924 6.26901Z"
        />
        <path
          strokeWidth="0.8"
          stroke="currentColor"
          strokeOpacity="0.306798"
          d="M10.8448 10.833V21.4209L19.6401 16.8753V6.26901L10.8448 10.833Z"
        />
        <path
          strokeWidth="0.8"
          stroke="currentColor"
          strokeOpacity="0.306798"
          d="M18.7569 4.90183L10.0384 0.39325L1.3392 4.90183L10.0384 9.42888L18.7569 4.90183Z"
        />
      </g>
      <defs>
        <clipPath id="clip0_67_62">
          <rect width="20" height="22" fill="currentColor" />
        </clipPath>
      </defs>
    </svg>
  );
}

export function IconSwap({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" className={cn('size-4', className)} {...props}>
      <rect fill="none" width="256" height="256" />
      <polyline
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        points="168 96 216 96 216 48"
      />
      <path
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M216,96,187.72,67.72A88,88,0,0,0,64,67"
      />
      <polyline
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        points="88 160 40 160 40 208"
      />
      <path
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M40,160l28.28,28.28A88,88,0,0,0,192,189"
      />
    </svg>
  );
}

export function IconInfo({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M128,24A104,104,0,1,0,232,128,104.11,104.11,0,0,0,128,24Zm0,192a88,88,0,1,1,88-88A88.1,88.1,0,0,1,128,216Zm16-40a8,8,0,0,1-8,8,16,16,0,0,1-16-16V128a8,8,0,0,1,0-16,16,16,0,0,1,16,16v40A8,8,0,0,1,144,176ZM112,84a12,12,0,1,1,12,12A12,12,0,0,1,112,84Z"></path>
    </svg>
  );
}

export function IconLogin({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M141.66,133.66l-40,40a8,8,0,0,1-11.32-11.32L116.69,136H24a8,8,0,0,1,0-16h92.69L90.34,93.66a8,8,0,0,1,11.32-11.32l40,40A8,8,0,0,1,141.66,133.66ZM200,32H136a8,8,0,0,0,0,16h56V208H136a8,8,0,0,0,0,16h64a8,8,0,0,0,8-8V40A8,8,0,0,0,200,32Z"></path>
    </svg>
  );
}

export function IconUpload({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" className={cn('size-4', className)} {...props}>
      <rect width="256" fill="none" height="256" />
      <line
        y2="32"
        x1="128"
        y1="144"
        x2="128"
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <polyline
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        points="216 144 216 208 40 208 40 144"
      />
      <polyline
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        points="88 72 128 32 168 72"
      />
    </svg>
  );
}

export function IconLogs({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 16 16" strokeLinejoin="round" className={cn('size-4', className)} {...props}>
      <path
        clipRule="evenodd"
        fillRule="evenodd"
        fill="currentColor"
        d="M9 2H9.75H14.25H15V3.5H14.25H9.75H9V2ZM9 12.5H9.75H14.25H15V14H14.25H9.75H9V12.5ZM9.75 7.25H9V8.75H9.75H14.25H15V7.25H14.25H9.75ZM1 12.5H1.75H2.25H3V14H2.25H1.75H1V12.5ZM1.75 2H1V3.5H1.75H2.25H3V2H2.25H1.75ZM1 7.25H1.75H2.25H3V8.75H2.25H1.75H1V7.25ZM5.75 12.5H5V14H5.75H6.25H7V12.5H6.25H5.75ZM5 2H5.75H6.25H7V3.5H6.25H5.75H5V2ZM5.75 7.25H5V8.75H5.75H6.25H7V7.25H6.25H5.75Z"
      ></path>
    </svg>
  );
}

export function IconLibrary({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="none"
      viewBox="0 0 40 40"
      xmlns="http://www.w3.org/2000/svg"
      className={cn(`size-4`, className)}
      {...props}
    >
      <path
        fill="url(#paint0_linear_495_530)"
        d="M35.1172 36.5469H21.9297C21.2344 36.5469 20.5547 36.3203 20 35.9063C19.4453 36.3281 18.7656 36.5547 18.0703 36.5469H4.88281C3 36.5469 1.21875 35.6797 0.046875 34.2031C0.375 36.6172 2.44531 38.4219 4.88281 38.4219H18.0703C19.1328 38.4219 20 37.5547 20 36.4922C20 37.5547 20.8672 38.4219 21.9297 38.4219H35.1172C37.5547 38.4219 39.625 36.6172 39.9531 34.2031C38.7891 35.6797 37.0078 36.5469 35.1172 36.5469Z"
      />
      <path
        fill="url(#paint1_linear_495_530)"
        d="M40 3.51562V30.375C40 33.0703 37.8125 35.2578 35.1172 35.2578H21.9297C20.8672 35.2578 20 34.3906 20 33.3281C20 34.3906 19.1328 35.2578 18.0703 35.2578H4.88281C2.1875 35.25 0 33.0703 0 30.375V3.51562C0 2.45312 0.867188 1.58594 1.92969 1.58594H13.3125C17.0078 1.58594 20 4.57813 20 8.27344C20 4.57813 22.9922 1.58594 26.6875 1.58594H38.0703C39.1328 1.58594 40 2.44531 40 3.51562Z"
      />
      <defs>
        <linearGradient
          y1="36.3733"
          y2="44.4771"
          x1="-18.1816"
          x2="-17.3249"
          id="paint0_linear_495_530"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#FBDA61" />
          <stop offset="1" stopColor="#FFC371" />
        </linearGradient>
        <linearGradient
          y1="18.9069"
          x2="13.9514"
          y2="57.1854"
          x1="-18.2713"
          id="paint1_linear_495_530"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#FBDA61" />
          <stop offset="1" stopColor="#FFC371" />
        </linearGradient>
      </defs>
    </svg>
  );
}

export function IconQuestion({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M140,180a12,12,0,1,1-12-12A12,12,0,0,1,140,180ZM128,72c-22.06,0-40,16.15-40,36v4a8,8,0,0,0,16,0v-4c0-11,10.77-20,24-20s24,9,24,20-10.77,20-24,20a8,8,0,0,0-8,8v8a8,8,0,0,0,16,0v-.72c18.24-3.35,32-17.9,32-35.28C168,88.15,150.06,72,128,72Zm104,56A104,104,0,1,1,128,24,104.11,104.11,0,0,1,232,128Zm-16,0a88,88,0,1,0-88,88A88.1,88.1,0,0,0,216,128Z"></path>
    </svg>
  );
}

export function IconLink({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M240,88.23a54.43,54.43,0,0,1-16,37L189.25,160a54.27,54.27,0,0,1-38.63,16h-.05A54.63,54.63,0,0,1,96,119.84a8,8,0,0,1,16,.45A38.62,38.62,0,0,0,150.58,160h0a38.39,38.39,0,0,0,27.31-11.31l34.75-34.75a38.63,38.63,0,0,0-54.63-54.63l-11,11A8,8,0,0,1,135.7,59l11-11A54.65,54.65,0,0,1,224,48,54.86,54.86,0,0,1,240,88.23ZM109,185.66l-11,11A38.41,38.41,0,0,1,70.6,208h0a38.63,38.63,0,0,1-27.29-65.94L78,107.31A38.63,38.63,0,0,1,144,135.71a8,8,0,0,0,16,.45A54.86,54.86,0,0,0,144,96a54.65,54.65,0,0,0-77.27,0L32,130.75A54.62,54.62,0,0,0,70.56,224h0a54.28,54.28,0,0,0,38.64-16l11-11A8,8,0,0,0,109,185.66Z"></path>
    </svg>
  );
}

export function IconPlay({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      width="15"
      height="15"
      fill="none"
      viewBox="0 0 15 15"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        fill="currentColor"
        d="M3.24182 2.32181C3.3919 2.23132 3.5784 2.22601 3.73338 2.30781L12.7334 7.05781C12.8974 7.14436 13 7.31457 13 7.5C13 7.68543 12.8974 7.85564 12.7334 7.94219L3.73338 12.6922C3.5784 12.774 3.3919 12.7687 3.24182 12.6782C3.09175 12.5877 3 12.4252 3 12.25V2.75C3 2.57476 3.09175 2.4123 3.24182 2.32181ZM4 3.57925V11.4207L11.4288 7.5L4 3.57925Z"
      ></path>
    </svg>
  );
}

export function IconPause({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" className={cn('size-4', className)} {...props}>
      <rect width="256" fill="none" height="256" />
      <rect
        y="40"
        rx="8"
        x="152"
        width="56"
        fill="none"
        height="176"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <rect
        x="48"
        y="40"
        rx="8"
        width="56"
        fill="none"
        height="176"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function IconExperiment({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      className={cn('size-4', className)}
      {...props}
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect width="16" height="16" fill="none" />
      <g mask="url(#mask0_707_393)">
        <path
          fill="currentColor"
          d="M3.33329 14C2.76663 14 2.36385 13.7472 2.12496 13.2417C1.88607 12.7361 1.9444 12.2667 2.29996 11.8333L5.99996 7.33333V3.33333H5.33329C5.1444 3.33333 4.98607 3.26944 4.85829 3.14167C4.73052 3.01389 4.66663 2.85556 4.66663 2.66667C4.66663 2.47778 4.73052 2.31944 4.85829 2.19167C4.98607 2.06389 5.1444 2 5.33329 2H10.6666C10.8555 2 11.0138 2.06389 11.1416 2.19167C11.2694 2.31944 11.3333 2.47778 11.3333 2.66667C11.3333 2.85556 11.2694 3.01389 11.1416 3.14167C11.0138 3.26944 10.8555 3.33333 10.6666 3.33333H9.99996V7.33333L13.7 11.8333C14.0555 12.2667 14.1138 12.7361 13.875 13.2417C13.6361 13.7472 13.2333 14 12.6666 14H3.33329ZM4.66663 12H11.3333L9.06663 9.33333H6.93329L4.66663 12ZM3.33329 12.6667H12.6666L8.66663 7.8V3.33333H7.33329V7.8L3.33329 12.6667Z"
        />
      </g>
    </svg>
  );
}

export function IconDotsThreeVertical({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      width="25"
      height="25"
      fill="none"
      viewBox="0 0 25 25"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path
        fill="currentColor"
        d="M10.9736 18.5C10.9736 19.6 11.8736 20.5 12.9736 20.5C14.0736 20.5 14.9736 19.6 14.9736 18.5C14.9736 17.4 14.0736 16.5 12.9736 16.5C11.8736 16.5 10.9736 17.4 10.9736 18.5ZM10.9736 6.5C10.9736 7.6 11.8736 8.5 12.9736 8.5C14.0736 8.5 14.9736 7.6 14.9736 6.5C14.9736 5.4 14.0736 4.5 12.9736 4.5C11.8736 4.5 10.9736 5.4 10.9736 6.5ZM10.9736 12.5C10.9736 13.6 11.8736 14.5 12.9736 14.5C14.0736 14.5 14.9736 13.6 14.9736 12.5C14.9736 11.4 14.0736 10.5 12.9736 10.5C11.8736 10.5 10.9736 11.4 10.9736 12.5Z"
      />
    </svg>
  );
}

export function IconPaperClip({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" className={cn('size-4', className)} {...props}>
      <rect width="256" fill="none" height="256" />
      <path
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M160,80,76.69,164.69a16,16,0,0,0,22.63,22.62L198.63,86.63a32,32,0,0,0-45.26-45.26L54.06,142.06a48,48,0,0,0,67.88,67.88L204,128"
      />
    </svg>
  );
}

export function IconFile({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" className={cn('size-4', className)} {...props}>
      <rect width="256" fill="none" height="256" />
      <path
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M200,224H56a8,8,0,0,1-8-8V40a8,8,0,0,1,8-8h96l56,56V216A8,8,0,0,1,200,224Z"
      />
      <polyline
        fill="none"
        strokeWidth="16"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        points="152 32 152 88 208 88"
      />
    </svg>
  );
}

export function IconLightBulb({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      className={cn('size-4', className)}
      {...props}
      fill="none"
      viewBox="0 0 10 14"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        fill="currentColor"
        d="M3.00004 13C3.00004 13.3666 3.30004 13.6666 3.66671 13.6666H6.33337C6.70004 13.6666 7.00004 13.3666 7.00004 13V12.3333H3.00004V13ZM5.00004 0.333313C2.42671 0.333313 0.333374 2.42665 0.333374 4.99998C0.333374 6.58665 1.12671 7.97998 2.33337 8.82665V10.3333C2.33337 10.7 2.63337 11 3.00004 11H7.00004C7.36671 11 7.66671 10.7 7.66671 10.3333V8.82665C8.87337 7.97998 9.66671 6.58665 9.66671 4.99998C9.66671 2.42665 7.57337 0.333313 5.00004 0.333313ZM6.90004 7.73331L6.33337 8.13331V9.66665H3.66671V8.13331L3.10004 7.73331C2.20004 7.10665 1.66671 6.08665 1.66671 4.99998C1.66671 3.15998 3.16004 1.66665 5.00004 1.66665C6.84004 1.66665 8.33337 3.15998 8.33337 4.99998C8.33337 6.08665 7.80004 7.10665 6.90004 7.73331Z"
      />
    </svg>
  );
}

export function IconZoomIn({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M152,112a8,8,0,0,1-8,8H120v24a8,8,0,0,1-16,0V120H80a8,8,0,0,1,0-16h24V80a8,8,0,0,1,16,0v24h24A8,8,0,0,1,152,112Zm77.66,117.66a8,8,0,0,1-11.32,0l-50.06-50.07a88.11,88.11,0,1,1,11.31-11.31l50.07,50.06A8,8,0,0,1,229.66,229.66ZM112,184a72,72,0,1,0-72-72A72.08,72.08,0,0,0,112,184Z"></path>
    </svg>
  );
}

export function IconZoomOut({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M152,112a8,8,0,0,1-8,8H80a8,8,0,0,1,0-16h64A8,8,0,0,1,152,112Zm77.66,117.66a8,8,0,0,1-11.32,0l-50.06-50.07a88.11,88.11,0,1,1,11.31-11.31l50.07,50.06A8,8,0,0,1,229.66,229.66ZM112,184a72,72,0,1,0-72-72A72.08,72.08,0,0,0,112,184Z"></path>
    </svg>
  );
}

export function IconSadFace({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
      fill="none"
    >
      <path
        fill="currentColor"
        d="M12 0C5.38321 0 0 5.38321 0 12C0 18.6168 5.38321 24 12 24C18.6168 24 24 18.6168 24 12C24 5.38321 18.6168 0 12 0ZM12 22.7027C9.0602 22.7027 6.39373 21.5111 4.45749 19.5859C3.68429 18.8171 3.02793 17.931 2.51687 16.9572C1.73856 15.4743 1.29731 13.7878 1.29731 12C1.29731 6.09852 6.09852 1.29731 12 1.29731C14.7992 1.29731 17.3504 2.37798 19.2595 4.14346C20.2494 5.05872 21.0668 6.15787 21.6562 7.38642C22.3267 8.7842 22.7027 10.349 22.7027 12C22.7027 17.9015 17.9015 22.7027 12 22.7027Z"
      />
      <path
        fill="currentColor"
        d="M13.2681 15.2116C14.6914 15.5092 15.9699 16.3094 16.8682 17.4648L17.8923 16.6686C16.8057 15.2707 15.2576 14.3023 13.5337 13.9417C10.7611 13.3623 7.84608 14.4324 6.1076 16.6686L7.13191 17.4648C8.56836 15.617 10.9771 14.7324 13.2681 15.2116Z"
      />
      <path
        fill="currentColor"
        d="M8.00006 10.6216C8.71655 10.6216 9.29737 10.0408 9.29737 9.32429C9.29737 8.60781 8.71655 8.02698 8.00006 8.02698C7.28357 8.02698 6.70274 8.60781 6.70274 9.32429C6.70274 10.0408 7.28357 10.6216 8.00006 10.6216Z"
      />
      <path
        fill="currentColor"
        d="M16.0272 10.6216C16.7437 10.6216 17.3245 10.0408 17.3245 9.32429C17.3245 8.60781 16.7437 8.02698 16.0272 8.02698C15.3107 8.02698 14.7299 8.60781 14.7299 9.32429C14.7299 10.0408 15.3107 10.6216 16.0272 10.6216Z"
      />
    </svg>
  );
}

export function IconMehFace({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
      fill="none"
    >
      <path
        fill="currentColor"
        d="M12 0C5.38321 0 0 5.38321 0 12C0 18.6168 5.38321 24 12 24C18.6168 24 24 18.6168 24 12C24 5.38321 18.6169 0 12 0ZM12 22.7027C9.0602 22.7027 6.39373 21.5111 4.45749 19.5859C3.68429 18.8171 3.02793 17.931 2.51687 16.9572C1.73856 15.4743 1.29731 13.7879 1.29731 12C1.29731 6.09852 6.09852 1.29731 12 1.29731C14.7992 1.29731 17.3504 2.37798 19.2595 4.14346C20.2494 5.05872 21.0668 6.15787 21.6562 7.3865C22.3267 8.7842 22.7027 10.349 22.7027 12C22.7027 17.9015 17.9015 22.7027 12 22.7027Z"
      />
      <path fill="currentColor" d="M18.4816 12.7444L5.2536 15.5007L5.51825 16.7708L18.7462 14.0144L18.4816 12.7444Z" />
      <path
        fill="currentColor"
        d="M7.98635 10.7027C8.70284 10.7027 9.28366 10.1219 9.28366 9.40538C9.28366 8.68889 8.70284 8.10806 7.98635 8.10806C7.26986 8.10806 6.68903 8.68889 6.68903 9.40538C6.68903 10.1219 7.26986 10.7027 7.98635 10.7027Z"
      />
      <path
        fill="currentColor"
        d="M16.0135 10.7027C16.73 10.7027 17.3108 10.1219 17.3108 9.40538C17.3108 8.68889 16.73 8.10806 16.0135 8.10806C15.297 8.10806 14.7162 8.68889 14.7162 9.40538C14.7162 10.1219 15.297 10.7027 16.0135 10.7027Z"
      />
    </svg>
  );
}

export function IconSmileFace({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
      fill="none"
    >
      <path
        fill="currentColor"
        d="M12 0C5.38321 0 0 5.38321 0 12C0 18.6168 5.38321 24 12 24C18.6168 24 24 18.6168 24 12C24 5.38321 18.6169 0 12 0ZM12 22.7027C9.06028 22.7027 6.39373 21.5111 4.45757 19.5859C3.68437 18.8171 3.02801 17.931 2.51695 16.9572C1.73856 15.4743 1.29731 13.7878 1.29731 12C1.29731 6.09852 6.09852 1.29731 12 1.29731C14.7992 1.29731 17.3504 2.37798 19.2595 4.14346C20.2494 5.05872 21.0668 6.15787 21.6562 7.38642C22.3267 8.7842 22.7027 10.349 22.7027 12C22.7027 17.9015 17.9015 22.7027 12 22.7027Z"
      />
      <path
        fill="currentColor"
        d="M8.08114 10.6216C8.79762 10.6216 9.37845 10.0408 9.37845 9.32429C9.37845 8.60781 8.79762 8.02698 8.08114 8.02698C7.36465 8.02698 6.78382 8.60781 6.78382 9.32429C6.78382 10.0408 7.36465 10.6216 8.08114 10.6216Z"
      />
      <path
        fill="currentColor"
        d="M16.1083 10.6216C16.8248 10.6216 17.4056 10.0408 17.4056 9.32429C17.4056 8.60781 16.8248 8.02698 16.1083 8.02698C15.3918 8.02698 14.811 8.60781 14.811 9.32429C14.811 10.0408 15.3918 10.6216 16.1083 10.6216Z"
      />
      <path
        fill="currentColor"
        d="M11.9771 18.6485C14.4869 18.6485 16.8922 17.365 18.2694 15.2339L17.1797 14.5298C15.9151 16.4867 13.6019 17.5788 11.2856 17.3115C9.48053 17.1034 7.81129 16.0634 6.82023 14.5298L5.73064 15.2339C6.92985 17.0898 8.9509 18.3482 11.1369 18.6003C11.4175 18.6327 11.6977 18.6485 11.9771 18.6485Z"
      />
    </svg>
  );
}

export function IconEye({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-4', className)}
      {...props}
    >
      <path d="M247.31,124.76c-.35-.79-8.82-19.58-27.65-38.41C194.57,61.26,162.88,48,128,48S61.43,61.26,36.34,86.35C17.51,105.18,9,124,8.69,124.76a8,8,0,0,0,0,6.5c.35.79,8.82,19.57,27.65,38.4C61.43,194.74,93.12,208,128,208s66.57-13.26,91.66-38.34c18.83-18.83,27.3-37.61,27.65-38.4A8,8,0,0,0,247.31,124.76ZM128,192c-30.78,0-57.67-11.19-79.93-33.25A133.47,133.47,0,0,1,25,128,133.33,133.33,0,0,1,48.07,97.25C70.33,75.19,97.22,64,128,64s57.67,11.19,79.93,33.25A133.46,133.46,0,0,1,231.05,128C223.84,141.46,192.43,192,128,192Zm0-112a48,48,0,1,0,48,48A48.05,48.05,0,0,0,128,80Zm0,80a32,32,0,1,1,32-32A32,32,0,0,1,128,160Z"></path>
    </svg>
  );
}

export function IconPen({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      className={cn('size-4', className)}
      {...props}
      fill="none"
      viewBox="0 0 16 16"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        fill="currentColor"
        d="M0.499023 15.5011H3.62402L12.8407 6.28444L9.71569 3.15944L0.499023 12.3761V15.5011ZM2.16569 13.0678L9.71569 5.51777L10.4824 6.28444L2.93236 13.8344H2.16569V13.0678Z"
      />
      <path
        fill="currentColor"
        d="M13.3074 0.742773C12.9824 0.417773 12.4574 0.417773 12.1324 0.742773L10.6074 2.26777L13.7324 5.39277L15.2574 3.86777C15.5824 3.54277 15.5824 3.01777 15.2574 2.69277L13.3074 0.742773Z"
      />
    </svg>
  );
}

export function IconCable({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      width="16"
      height="16"
      fill="none"
      className={cn('size-4', className)}
      {...props}
      viewBox="0 0 16 16"
      xmlns="http://www.w3.org/2000/svg"
    >
      <mask x="0" y="0" width="16" height="16" id="mask0_758_1331" maskUnits="userSpaceOnUse">
        <rect width="16" height="16" fill="currentColor" />
      </mask>
      <g mask="url(#mask0_758_1331)">
        <path
          fill="currentColor"
          d="M3.33333 14C3.14444 14 2.98611 13.9361 2.85833 13.8083C2.73056 13.6806 2.66667 13.5222 2.66667 13.3333V12.6667H2V10C2 9.81111 2.06389 9.65278 2.19167 9.525C2.31944 9.39722 2.47778 9.33333 2.66667 9.33333H3.33333V4.66667C3.33333 3.93333 3.59444 3.30556 4.11667 2.78333C4.63889 2.26111 5.26667 2 6 2C6.73333 2 7.36111 2.26111 7.88333 2.78333C8.40556 3.30556 8.66667 3.93333 8.66667 4.66667V11.3333C8.66667 11.7 8.79722 12.0139 9.05833 12.275C9.31945 12.5361 9.63333 12.6667 10 12.6667C10.3667 12.6667 10.6806 12.5361 10.9417 12.275C11.2028 12.0139 11.3333 11.7 11.3333 11.3333V6.66667H10.6667C10.4778 6.66667 10.3194 6.60278 10.1917 6.475C10.0639 6.34722 10 6.18889 10 6V3.33333H10.6667V2.66667C10.6667 2.47778 10.7306 2.31944 10.8583 2.19167C10.9861 2.06389 11.1444 2 11.3333 2H12.6667C12.8556 2 13.0139 2.06389 13.1417 2.19167C13.2694 2.31944 13.3333 2.47778 13.3333 2.66667V3.33333H14V6C14 6.18889 13.9361 6.34722 13.8083 6.475C13.6806 6.60278 13.5222 6.66667 13.3333 6.66667H12.6667V11.3333C12.6667 12.0667 12.4056 12.6944 11.8833 13.2167C11.3611 13.7389 10.7333 14 10 14C9.26667 14 8.63889 13.7389 8.11667 13.2167C7.59444 12.6944 7.33333 12.0667 7.33333 11.3333V4.66667C7.33333 4.3 7.20278 3.98611 6.94167 3.725C6.68056 3.46389 6.36667 3.33333 6 3.33333C5.63333 3.33333 5.31944 3.46389 5.05833 3.725C4.79722 3.98611 4.66667 4.3 4.66667 4.66667V9.33333H5.33333C5.52222 9.33333 5.68056 9.39722 5.80833 9.525C5.93611 9.65278 6 9.81111 6 10V12.6667H5.33333V13.3333C5.33333 13.5222 5.26944 13.6806 5.14167 13.8083C5.01389 13.9361 4.85556 14 4.66667 14H3.33333Z"
        />
      </g>
    </svg>
  );
}
