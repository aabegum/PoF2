# Healthy Equipment Data Requirements
**Quick Reference Guide**

## What You Need to Provide

**File Name**: `data/healthy_equipment.csv`

**Required Columns**:
```csv
cbs_id,Şebeke Unsuru,Sebekeye_Baglanma_Tarihi
50001,Ayırıcı,2015-03-20
50002,Kesici,2018-07-12
50003,OG/AG Trafo,2012-11-05
50004,AG Anahtar,2016-09-15
50005,Rekortman,2014-06-30
```

**Optional Columns** (recommended for better accuracy):
- `Enlem` (Latitude)
- `Boylam` (Longitude)
- `Musteri_Sayisi` (Customer count)
- `Trafo_Güç_kVA` (Transformer capacity)

## Sample Size Recommendation

**Current failed equipment**: ~1,313
**Target healthy equipment**: 1,300 - 2,600

**Ratio Guidelines**:
- 1:1 ratio (1,300 healthy) → Balanced training
- 2:1 ratio (2,600 healthy) → More realistic distribution

## Data Quality Checklist

- [ ] **IDs don't overlap** with failed equipment
- [ ] **Equipment types** match categories in `Şebeke Unsuru` column
- [ ] **Dates** in `YYYY-MM-DD` or `DD-MM-YYYY` format
- [ ] **Completeness**: At least 80% of rows have required columns
- [ ] **Representative**: Equipment types match distribution of failed equipment

## Definition of "Healthy"

**Recommended**:
- Equipment with **zero failures** in last 12 months, AND
- Equipment with **< 2 lifetime failures**

**Alternative** (if above not available):
- Equipment with **zero failures ever**
- Equipment currently **active** in the grid

## Example Data

```csv
cbs_id,Şebeke Unsuru,Sebekeye_Baglanma_Tarihi,Enlem,Boylam,Musteri_Sayisi
50001,Ayırıcı,2015-03-20,41.0082,28.9784,150
50002,Kesici,2018-07-12,41.0123,29.0012,200
50003,OG/AG Trafo,2012-11-05,40.9876,28.8956,500
50004,AG Anahtar,2016-09-15,41.0234,28.9123,80
50005,Rekortman,2014-06-30,40.9988,29.0234,120
50006,OG Hat,2013-04-18,41.0045,28.9567,300
50007,AG Pano,2017-02-22,41.0190,29.0089,95
```

## Where to Place File

```
PoF2/
├── data/
│   ├── combined_data_son.xlsx          (existing - fault data)
│   └── healthy_equipment.csv           (NEW - you provide this)
```

## Equipment Type Categories

Make sure your `Şebeke Unsuru` values match these categories:
- Ayırıcı (Disconnector)
- Kesici (Circuit Breaker)
- OG/AG Trafo (Transformer)
- AG Anahtar (LV Switch)
- AG Pano Box (LV Panel Box)
- Bina (Building/Substation)
- Rekortman (Recloser)
- OG Hat (MV Line)
- AG Hat (LV Line)
- AG Pano (LV Panel)
- Trafo Bina Tip (Building-type Transformer)

## Questions?

1. **Where does healthy equipment data come from?**
   - CBS system query for equipment with no recent failures
   - Manual selection from active equipment inventory

2. **Should I match equipment type distribution?**
   - **Yes, recommended** - helps model learn patterns for each type

3. **What if I can't get 1,300 healthy equipment?**
   - Minimum: 500 healthy equipment (1:2.6 ratio)
   - Below this, class imbalance becomes problematic

4. **Can healthy equipment have old failures?**
   - **Yes** - Equipment with 1 old failure (>24 months ago) can be included
   - The model will learn to distinguish "recovered" vs "chronic" equipment

## Once You Provide Data

The pipeline will:
1. ✅ Load healthy equipment
2. ✅ Merge with failed equipment dataset
3. ✅ Create balanced training data (positive + negative samples)
4. ✅ Train models to distinguish healthy vs at-risk equipment
5. ✅ Generate improved risk scores with better calibration

---

**Ready?** → Place `data/healthy_equipment.csv` and let me know!
