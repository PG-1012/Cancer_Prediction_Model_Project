### Hey, I'm Panshul 👋

Working in fintech in Toronto 🇨🇦

I like problems where the interesting part isn't the code — it's figuring out
what the right question was. Usually that means digging until the actual
constraint shows up, then building the smallest thing that respects it.

---

### Things I've built

**🏛️ [Equity Research Tool](https://github.com/PG-1012/Equity_Research_Tool)** · Python · Streamlit
> Congressional trade disclosures, live market data and LLM analysis in one workbench.
>
> The House Clerk publishes STOCK Act filings only as PDFs — no ruling lines, and a
> text layer that extracts as mojibake. Turned out to be a small-caps font mapped at a
> fixed `+0x222` Unicode offset, so it inverts exactly. **7,667 transactions parsed,
> 99.5% passing validation.** Trades plot straight onto the price chart.

**⌨️ [FlowKeys](https://github.com/PG-1012/FlowKeys)** · Swift · macOS
> Copy several things, paste the one you want. Hold ⌘, tap V to walk back through
> clipboard history.
>
> Modelled on ⌘Tab: a quick tap pastes normally, so ordinary paste is untouched.
> Needed a `CGEventTap` that can actually *consume* the keystroke — a global monitor
> can watch ⌘V go by but never stop it.

**🔬 [Breast Cytology Classifier](https://github.com/PG-1012/Cancer_Prediction_Model_Project)** · Python · PyTorch
> Benign vs malignant from nine cytological attributes — and which mistake we're
> actually trying to avoid.
>
> Every model lands near 96%. The real lever was the decision threshold: moving it
> off the default 0.5 took missed malignancies from **12 to 2** for three extra false
> alarms. The neural net never beat logistic regression.

---

### Working on

`Python` `Swift` `TypeScript` `PyTorch` `scikit-learn` `Streamlit`

Currently curious about how much of ML engineering is really just choosing the
right loss to care about.

📫 [rdp.gera@gmail.com](mailto:rdp.gera@gmail.com)
