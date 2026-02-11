import * as React from "react"
import { cva } from "class-variance-authority"
import { cn } from "@/lib/utils"

/**
 * Ink & Ember Badge
 * - Fully token-driven
 * - Theme-agnostic
 * - Radius controlled by --radius (Void = sharp)
 * - No SaaS gloss
 */

const badgeVariants = cva(
  [
    "inline-flex items-center",
    "border",
    "px-2.5 py-0.5",
    "text-xs font-semibold tracking-wide",
    "select-none whitespace-nowrap",
    "transition-colors",
    "focus:outline-none",
    "focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:ring-offset-background",
    // Radius comes from CSS variable (Void theme sets 0rem)
    "rounded-[var(--radius)]",
  ].join(" "),
  {
    variants: {
      variant: {
        // Primary “seal”
        default:
          "bg-primary text-primary-foreground border-transparent",

        // Quiet catalog/meta tag
        muted:
          "bg-secondary text-secondary-foreground border-border",

        // Artifact label style
        outline:
          "bg-transparent text-foreground border-border",

        // Status
        danger:
          "bg-destructive text-destructive-foreground border-transparent",

        warning:
          "bg-warning text-warning-foreground border-transparent",

        success:
          "bg-success text-success-foreground border-transparent",
      },
      size: {
        sm: "px-2 py-0 text-[11px]",
        md: "px-2.5 py-0.5 text-xs",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "md",
    },
  }
)

function Badge({ className, variant, size, ...props }) {
  return (
    <span
      data-testid={props["data-testid"] || "badge"}
      className={cn(badgeVariants({ variant, size }), className)}
      {...props}
    />
  )
}

export { Badge, badgeVariants }
