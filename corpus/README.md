# Demo corpus

Eight public documents chosen to exercise every parser and both languages: English security guidance
from NIST and OWASP, the Russian personal data law, and two scans that only OCR can read. The theme is
one a real team searches: passwords, authentication and personal data.

| File | Format | What it exercises | Source | Licence |
| --- | --- | --- | --- | --- |
| `nist-sp-800-63b-4.pdf` | PDF, 129 pages | Born-digital text, PDF bookmarks, numbered headings, running headers, hyphenated line breaks | [NIST SP 800-63B-4](https://doi.org/10.6028/NIST.SP.800-63B-4) (2025) | U.S. government work |
| `nist-csf-2-0.pdf` | PDF, 32 pages | Tables, appendices, glossary | [NIST CSF 2.0](https://doi.org/10.6028/NIST.CSWP.29) (2024) | U.S. government work |
| `nist-sp-800-53r5-controls.xlsx` | Excel, 1,189 rows | One record per row, cited by sheet and row; long cells split with the control id kept | [SP 800-53 Rev. 5 control catalog](https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final) | U.S. government work |
| `fips-112-password-usage-1985.pdf` | Scanned PDF, 60 pages | OCR (`"ocr": "always"`: the scan's own text layer is older and noisier) | [FIPS PUB 112](https://nvlpubs.nist.gov/nistpubs/Legacy/FIPS/fipspub112.pdf) (1985) | U.S. government work |
| `owasp-password-storage-cheat-sheet.md` | Markdown | Heading hierarchy, code blocks, tables, inline markup | [OWASP Cheat Sheet Series](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html) | CC BY-SA 4.0 |
| `owasp-authentication-cheat-sheet.md` | Markdown | As above | [OWASP Cheat Sheet Series](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html) | CC BY-SA 4.0 |
| `ru-152-fz-personal-data.docx` | Word | Heading styles (chapters and articles), Russian morphology | [Федеральный закон № 152-ФЗ](http://pravo.gov.ru/proxy/ips/?docbody=&nd=102108261), current consolidated text | Official document, not subject to copyright (art. 1259 of the Civil Code of the Russian Federation) |
| `ru-519-fz-2020-scan.pdf` | Scanned PDF, 8 pages | Russian OCR of an image-only PDF | [Федеральный закон от 30.12.2020 № 519-ФЗ](http://publication.pravo.gov.ru/Document/View/0001202012300044), official publication | Official document, not subject to copyright |

U.S. government works are not subject to copyright in the United States (17 U.S.C. § 105). The OWASP
cheat sheets are included unmodified under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
and attributed to the OWASP Cheat Sheet Series contributors.

## How the files were made

All files are downloaded unchanged except one. The official legal information system publishes 152-FZ
as HTML, so `scripts/fetch.py` converts it to Word: chapters become *Heading 1*, articles *Heading 2*,
every other paragraph is kept as is, and superscript article numbers become decimals ("Статья 10<sup>1</sup>"
is cited as "статья 10.1"). The conversion is deterministic, so the file matches its recorded checksum.

```bash
python corpus/scripts/fetch.py            # download everything again and rebuild the Word file
python corpus/scripts/fetch.py --verify   # check the committed files against manifest.json (CI does this)
```

Some Russian government sites refuse connections from outside Russia, which is why the files are
committed rather than downloaded in CI.

`manifest.json` lists every file with its id, title, source, licence, OCR mode and SHA-256;
`lumen ingest --manifest corpus/manifest.json` indexes them all.
