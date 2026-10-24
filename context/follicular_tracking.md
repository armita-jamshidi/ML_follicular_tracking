# Follicular tracking and the three classes

Background for interpreting results. Educational summary, not clinical guidance.

## What follicular tracking is

Follicular tracking (folliculometry) is a series of transvaginal ultrasound scans across a
menstrual cycle to watch ovarian follicles grow. It is used in fertility care to time
ovulation, intercourse, insemination or egg retrieval, and to check that ovulation happens.

A typical sequence:
1. **Early follicular phase.** Several small antral follicles (a few mm) are visible on
   each ovary.
2. **Selection.** Around the middle of the follicular phase one follicle outgrows the rest
   and becomes the **dominant follicle**, while the others regress.
3. **Growth.** The dominant follicle grows roughly 1–2 mm per day and usually reaches about
   18–25 mm before ovulation.
4. **Ovulation.** The follicle ruptures; afterwards a corpus luteum and some free fluid may
   be seen.

## How the dataset classes map onto this

| Class | What the image typically shows | Where it sits in tracking |
|---|---|---|
| Normal | Ovary with a few small follicles, none dominant | Early cycle, or a non-stimulated ovary |
| Dominant Follicle | One large, round, anechoic (dark) follicle standing out | Mid-to-late follicular phase |
| PCO | Many small follicles, often arranged around the edge ("string of pearls"), enlarged ovary | A morphology that disrupts normal selection |

A single image is one snapshot, so a model classifying it is answering "what does this
ovary look like now", not tracking a cycle. Tracking would need serial images from the same
person, which neither dataset has.

## PCO morphology is not PCOS

Polycystic ovarian **morphology** (PCOM) is an ultrasound finding. Polycystic ovary
**syndrome** (PCOS) is a clinical diagnosis that, under the Rotterdam criteria, needs two of
three: irregular or absent ovulation, clinical or biochemical hyperandrogenism, and PCOM,
after excluding other causes. PCOM is also common in people without PCOS. The 2018
international guideline raised the follicle-count threshold for PCOM to 20 or more follicles
per ovary with modern high-frequency transducers.

So a "PCO" prediction here means "this image resembles images a radiologist labelled PCO".
It never means a person has PCOS.

## Why the classes are hard to separate

- A Normal ovary early in the cycle and a PCO ovary both show multiple small follicles;
  the difference is count, size distribution, arrangement and ovarian volume, which a
  single 2D slice captures only partially.
- A large dominant follicle can coexist with PCO morphology.
- Image quality, gain settings and probe angle vary by operator and machine, which is why
  the acquisition audit matters.
