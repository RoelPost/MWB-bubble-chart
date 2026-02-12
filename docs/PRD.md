# Product Requirements Document
## MWB Construction Machinery - Activity Bubble Chart

### Overview
Interactive D3.js bubble chart visualization showing how construction machinery distributes their workday across different power states. Designed for LinkedIn engagement and embedded use in existing websites.

### Goal
Create a visually compelling, interactive web component that visualizes construction equipment usage patterns throughout a 24-hour day, building on previous MWB articles about machine hours and fuel consumption.

### Core Functionality
- **Interactive bubble chart** with gravity field effect (inspired by FlowingData's American Workday visualization)
- Shows multiple machine types simultaneously
- Each bubble represents activity periods in specific power states
- Bubbles move along 24-hour timeline with smooth transitions
- Hover interactions for detailed information

### Visual Specification

**Power States **
1. **Uitgeschakeld** (Off) 
2. **Stationair**  (idle)
3. **Laag vermogen** (Low power) 
4. **Hoog vermogen** (High power) 


**Machine Types (x-axis lanes)**
- Graafmachines (Excavators)
- Wielladers (Wheel loaders)
- Dumpers
- Kranen (Cranes)
- Verrijkers

**NOx**
emission - (color-coded)


### Data Format

**Input Requirements:** CSV file with minute-resolution time series

```csv
machine_id,machine_type,timestamp,power_state,nitrogen_emission
M001,Graafmachine,2024-01-15T06:00:00,off,0
M001,Graafmachine,2024-01-15T06:01:00,idle,2.3
M001,Graafmachine,2024-01-15T06:02:00,high,8.5
```

**Columns:**
- `machine_id`: Unique machine identifier
- `machine_type`: Graafmachine, Wiellader, Dumper, Kraan
- `timestamp`: ISO8601 format (YYYY-MM-DDTHH:MM:SS)
- `power_state`: off | idle | low | high
- `nitrogen_emission`: Numeric value

### Use Cases
1. **LinkedIn Content** - Eye-catching visualization for social engagement
2. **Efficiency Analysis** - Show idle vs. productive usage patterns
3. **Electrification Opportunities** - Identify idle periods suitable for electric alternatives
4. **Emission Insights** - Visualize emissions by power state

### Technical Requirements
- **Framework:** D3.js (v7+)
- **Output:** Embeddable HTML/JS component
- **Browser Support:** Modern browsers (Chrome, Firefox, Safari, Edge)
- **Responsive:** Adaptable to different screen sizes
- **Performance:** Smooth animations even with large datasets

### Reference Implementation
**Primary Reference:** FlowingData moving bubbles tutorial
- Location: `/Users/roelpost/Library/CloudStorage/OneDrive-LogisticsDesignLab/Research/Courses & Tutorials/Flowing Data/moving-bubbles-tutorial`
- Style: Moving bubble chart with gravity field (like [American Workday visualization](https://flowingdata.com/2017/05/17/american-workday/))
- This tutorial contains the core implementation pattern to replicate

**Key Features to Implement:**
- Force simulation with gravity centers for each machine type lane
- Color-coded bubbles by power state
- Time-based animation showing progression through the day

### Deliverable
Single-page web component that can be:
- Embedded in existing websites via iframe or direct inclusion
- Shared as standalone page
- Used in LinkedIn posts (screenshot/video)

### Out of Scope (Phase 1)
- Real-time data streaming
- Backend API development
- User authentication
- Data export functionality
- Multi-day comparisons

### Success Criteria
- Visually clear representation of machine usage patterns
- Smooth, engaging animations
- Easy to embed in existing web infrastructure
- Generates discussion on efficiency and sustainability
