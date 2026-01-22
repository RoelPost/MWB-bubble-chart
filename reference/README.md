# Reference Materials

## FlowingData Moving Bubbles Tutorial

This directory contains the reference implementation from the FlowingData tutorial that serves as the inspiration for our MWB bubble chart visualization.

### Tutorial Files

- **Make a Moving Bubbles Chart to Show Clustering and Distributions FlowingData.pdf** - Complete tutorial PDF
- **index.html** - Final complete implementation
- **index00.html** - Step 0: Initial setup
- **index01.html** - Step 1: Basic structure
- **index02.html** - Step 2: Adding force simulation
- **index03.html** - Step 3: Final version with interactions
- **js/** - JavaScript files
- **style/** - CSS files

### Key Concepts to Adapt

1. **Force Simulation** - Uses D3's force layout with:
   - `forceX()` for horizontal positioning (time-based in our case)
   - `forceY()` for vertical positioning (machine type lanes)
   - `forceCollide()` for bubble collision detection

2. **Gravity Centers** - Each machine type has its own "gravity center" that pulls bubbles to their lane

3. **Smooth Transitions** - Bubbles animate smoothly between positions as they move along the timeline

4. **Color Coding** - Different states/categories shown through color (power states in our implementation)

### How Our Implementation Differs

- **Data source**: CSV instead of JSON
- **X-axis**: 24-hour timeline instead of categorical
- **Y-axis**: Machine types (Graafmachine, Wiellader, Dumper, Kraan)
- **Colors**: Power states (off, idle, low, high) instead of activity types
- **Bubble size**: Based on nitrogen emissions

### Usage

Open any of the HTML files directly in a browser to see the tutorial examples in action.
