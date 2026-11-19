# Using real Northern Territory data

The default area is **East Tennant** (MGA53 E 520–620 km, N 7,780–7,880 km, roughly 135.2–136.1°E, 19.2–20.1°S). Geoscience Australia's Exploring for the Future program targeted this covered IOCG belt with new gravity, magnetotelluric and stratigraphic drilling data. Put the files under `data/raw/real/` with the names below, or edit `config/config.yaml`.

| Layer | Source | How to get it | Config key |
|---|---|---|---|
| Gravity (spherical cap Bouguer anomaly) | GA, *East Tennant Gravity Survey P201901* (2,552 stations infilling to 2 km spacing). NetCDF on the NCI THREDDS server | **Automatic**: the URL is already in `config.yaml` and `make run-real` downloads it | `sources.real.gravity` |
| Magnetics (TMI RTP) | GA, *Magnetic Anomaly Map of Australia, 7th edition* (PID 144725), also as derivative GeoTIFFs | Download from the GA Data Portal / eCat, or cut a bbox with THREDDS NCSS. Save as `mag_rtp.tif` | `sources.real.magnetics` |
| Surface geochemistry | NTGS **DIP001** *Northern Territory geochemical and drilling datasets* (GEMIS, about 1.1 GB zip, updated twice a year), or a STRIKE export for a smaller area | Download, filter to soil and rock-chip samples inside the bbox, save as `geochem.csv`, then map the column names and units under `geochem.columns` | `sources.real.geochem` |
| Known deposits / occurrences | NTGS mineral occurrence layer (STRIKE) or GA OZMIN | Export as CSV with lon/lat and commodity, save as `deposits.csv` | `sources.real.deposits` |

Links: STRIKE https://strike.nt.gov.au · GEMIS https://geoscience.nt.gov.au/gemis · GA eCat https://ecat.ga.gov.au · NCI THREDDS https://thredds.nci.org.au

## Checklist when switching to real data
1. **CRS.** Legacy NT points are usually GDA94 (EPSG:4283). Set `crs` per source. The ETL handles the GDA94 to GDA2020 shift (about 1.8 m) through PyProj.
2. **Below detection.** Check how the file codes it (negative numbers, `<2`, or a flag column) and adapt `etl/points.py` if it isn't negative-DL.
3. **Units.** Au is usually ppb and Fe is often %. Set `unit` per column (ppm, ppb, pct).
4. **Sample type.** Don't mix drill assays with soils without a `SAMPLE_TYPE` filter, because they have different backgrounds.
5. **Deposit filter.** Keep only Cu/Au occurrences that match your mineral system (`commodity_filter`).
6. **Grid size.** 500 m suits a 10,000 km² regional study. Go to 100–200 m for camp scale, which is roughly 25× more pixels.
7. **Re-validate.** Real data will score lower than the synthetic benchmark. That's expected, and it's the number to put on your CV.
