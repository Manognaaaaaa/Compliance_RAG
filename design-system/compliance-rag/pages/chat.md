# Chat Page Overrides

> Overrides `../MASTER.md` for the chat page (`Frontend/index.html`). Anything not listed here follows MASTER.md.
> Revised with the Taste Skill redesign protocol ("redesign - preserve"): regulated / trust-first tool, dials VARIANCE 3, MOTION 3, DENSITY 4.

## Tokens changed or added

| Token | Light | Dark | Why |
|-------|-------|------|-----|
| `--color-accent` | `#9E531A` | `#AD5C1F` | MASTER's `#B45309` is 90% saturated; capped below 80%. White icon on it: 5.67:1 light / 4.85:1 dark. |
| `--color-card` | `#FCFDFE` | `#111A2E` | No pure `#FFFFFF` surfaces. |
| `--color-border-strong` | `#64748B` | `#6B7A99` | Input boundary needs 3:1 against the page (WCAG 1.4.11); `--color-border` is 1.42:1. |
| `--color-user-bubble` | `#1E3A8A` | `#1E3A8A` | Split from `--color-primary`, which becomes light blue in dark mode for text/icons/focus. |
| Shadows | slate-tinted | near-black navy | Tinted to the palette, never pure black. |

Full dark palette: background `#0B1220`, foreground `#E6EAF2`, muted `#1A2540`, muted foreground `#A3AEC2`, border `#26324D`, primary `#9DB2EC`, destructive `#F87171`. All text pairs pass 4.5:1, all control boundaries pass 3:1.

## Component rules

- **Theme:** light and dark, following `prefers-color-scheme`. Test both before shipping.
- **Shape scale:** containers 16px, inner cards / buttons / icon tiles 12px (avatar 10px), chips full pill. No other radii.
- **Assistant answers:** no card chrome. Text sits on the page with a 2px left rule, max 68ch per line.
- **Starter questions:** muted tiles at rest, elevation only on hover. Press state scales to 0.985.
- **Error message:** `--color-destructive` text and left rule. No tinted red background (`#DC2626` on `#FEF2F2` is 4.41:1).
- **Input focus:** the ring goes on the whole input bar (`:focus-within`), not the textarea, so it isn't clipped.
- **Copy:** no em-dashes or en-dashes in UI strings. Source chips read "Document, p. N".
- **Page pattern:** MASTER's "Trust & Authority + Conversion" landing-page sections and sales CTAs don't apply. This page is a single-purpose chat tool: empty state (headline + starter questions) → conversation → input bar with disclaimer.
