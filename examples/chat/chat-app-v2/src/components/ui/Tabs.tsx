'use client';

import * as React from 'react';

import * as TabsPrimitive from '@radix-ui/react-tabs';

import { cn } from '@/lib/utils';

const TabsRoot = TabsPrimitive.Root;

const TabsList = TabsPrimitive.List;

const TabsTrigger = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Trigger
    ref={ref}
    className={cn(
      `
        border-b-2 border-b-transparent p-4 text-sm font-medium text-white

        data-[state='active']:border-green-500 data-[state='active']:text-green-500

        hover:text-green-500
      `,
      className,
    )}
    {...props}
  />
));
TabsTrigger.displayName = TabsPrimitive.Trigger.displayName;

const TabsContent = TabsPrimitive.Content;

export { TabsContent, TabsList, TabsRoot, TabsTrigger };
