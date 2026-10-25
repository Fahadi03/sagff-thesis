# Draft: data redistribution request to the GAN-BnBERT authors

Status: **draft, not sent.** Fill in the bracketed fields, find the authors' current
addresses (paper / ResearchGate / GitHub profile), and send from your university account.

---

**To:** Mehedi Hasan; Mohammad Jahid Ibna Basher; Md. Tanvir Rouf Shawon
**Subject:** Permission to redistribute derived splits of BNIntent30

Dear Mehedi Hasan, Mohammad Jahid Ibna Basher and Md. Tanvir Rouf Shawon,

I am [your name], a [degree] student at [university], working under the
supervision of [supervisor's name]. My thesis studies when handcrafted features
help low-resource intent classification in English and Bangla. BNIntent30, from
your paper "Bengali Intent Classification with Generative Adversarial BERT"
(ICCIT 2023, doi:10.1109/ICCIT60459.2023.10440989; arXiv:2312.10679), is the core dataset: because it has the English source and
the Bangla translation for every utterance, it allows a parallel English-vs-Bangla
comparison that no other dataset I know of supports. Thank you for releasing it.

The repository (github.com/JahidBasher/GAN-BnBERT, commit 213bbcf) has no license
file, and I could not find a data license in the paper. So for now my code
downloads the data directly from your repository and publishes only row
identifiers and checksums, never the text.

For reproducibility I would like to ask whether you would allow me to
redistribute **derived splits** of BNIntent30 (the utterances, intent labels and
my train/validation/test assignment, with full credit to your paper) alongside
my code and paper. If you would prefer a particular license (for example
CC BY 4.0 or CC BY-NC 4.0) or specific attribution wording, I will follow it
exactly. If you would rather I didn't redistribute the data, that is completely
fine and I will keep the current download-only approach.

You may also find one finding from my preprocessing useful. After removing the
67 `TRANSLATION_FAILED` rows (which reproduces your published 4,433-row count
exactly), some test utterances have Bangla text identical to training
utterances. This is because machine translation maps different English sentences
onto the same Bangla sentence. I would be glad to share the exact counts and the
script that finds them.

Thank you for your time, and for making the dataset available.

Kind regards,
[your name]
[degree, department, university]
[email address]
