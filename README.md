# MWB Bubble Chart - Bouwmachines Visualisatie

Interactieve bubble chart visualisatie van bouwmachines activiteit en NOx-uitstoot gedurende een werkdag.

## Live Visualisatie

Open `index.html` in een browser (via een lokale server) om de animated bubble chart te bekijken.

```bash
python3 -m http.server 8080
# Open http://localhost:8080
```

## Wat doet de visualisatie?

De bubble chart toont 84 bouwmachines die gedurende een werkdag (02:00 - 22:00) bewegen tussen vier toestanden:
- **Hoge belasting** - Werkend met motorbelasting >= 25%
- **Lage belasting** - Werkend met motorbelasting < 25%
- **Stationair** - Motor draait, machine staat stil
- **Uit** - Machine is uitgeschakeld

De kleur van elke bubble geeft de NOx-uitstoot weer (grijs = geen, geel→rood = laag→hoog).

## Bestanden

```
MWB bubble chart/
├── index.html                      # Hoofdvisualisatie (D3.js)
├── src/style.css                   # Styling
├── prepare_bubble_data.py          # Data preparation script
├── data/
│   ├── NOx_intervals - *.csv       # Brondata (10-min intervallen)
│   └── NOx_intervals_with_belasting.csv  # Verwerkte data
├── data_exploration_v2.ipynb       # Data analyse notebook
└── reference/                      # Tutorial en voorbeelden
```

## Data Preparation

Het script `prepare_bubble_data.py` transformeert de brondata:

1. **Belasting categorieën** - Bepaalt machine status op basis van:
   - `machine_staat` (Uit/Stationair/Werkend)
   - `motorbelasting` (drempel: 25%)

2. **Machine categorieën** - Groepeert 19 machine types naar 11 categorieën:
   | Categorie | Bevat |
   |-----------|-------|
   | Rupsgraafmachine | Hydraulische rupsgraafmachine |
   | Mobiele graafmachine | Mobiele graafmachine |
   | Lader | Lader |
   | Asfaltverwerking | Asfaltverwerking |
   | Asfaltverdichting | Asfaltverdichting |
   | Hijskraan | Mobiele/Rups/Vaste hijskraan |
   | Generator | Generatoren |
   | Grondverzet | Bulldozer, Dumper, Grondwals |
   | Heistelling | Heistelling, Heischip |
   | Tractor | Tractor, Werktuigdrager, Maaier |
   | Overig | Betonverwerking, Markeeringsmachine, Testopstelling |

3. **Representatieve dag** - Combineert data van meerdere dagen tot één geanimeerde dag (alle machines tegelijk zichtbaar)

### Data opnieuw genereren

```bash
python3 prepare_bubble_data.py
```

## Data Analyse

Open `data_exploration_v2.ipynb` voor:
- Overzicht machines per categorie
- Data coverage per machine
- Status patronen over tijd
- Motorbelasting over tijd
- Dagpatronen (per uur)
- NOx uitstoot over tijd

## Technologie

- **D3.js v7** - Visualisatie en animatie
- **Python/Pandas** - Data preparation
- **Jupyter** - Data exploratie
