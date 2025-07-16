## Overview
The script `import_hcm_docs.py` is designed to read the HCM documents, split them into chunks, generate embeddings, and store them in a ChromaDB collection.

For local usage of this mcp server, you need to have the HCM documents in the `data/hcm_files` directory. And make sure to split it into each chapter as a separate PDF file. I used 128 texts as chunk size for RAG usage.

It takes approximately <=30 minutes to embed and store all HCM documents in the ChromaDB database.

## Content

This is what separated files expected to be within this folder from the HCM documents and associated pages.

| Chapter | Page Range  | Title                                                 |
|---------|-------------|-------------------------------------------------------|
| Misc    | 1-146       | Preface                                               |
| 1       | 147–180     | HCM User's Guide                                      |
| 2       | 181–210     | Applications                                          |
| 3       | 211–277     | Modal Characteristics                                 |
| 4       | 278–353     | Traffic Operations and Capacity Concepts              |
| 5       | 354–377     | Quality and Level-of-Service Concepts                 |
| 6       | 378–428     | HCM and Alternative Analysis Tools                    |
| 7       | 429–490     | Interpreting HCM and Alternative Tool Results         |
| 8       | 491–526     | HCM Primer                                            |
| 9       | 527–625     | Glossary and Symbols                                  |
| 10      | 626–722     | Freeway Facilities Core Methodology                   |
| 11      | 723–800     | Freeway Reliability Analysis                          |
| 12      | 801–902     | Basic Freeway and Multilane Highway Segments          |
| 13      | 903–969     | Freeway Weaving Segments                              |
| 14      | 970–1047    | Freeway Merge and Diverge Segments                    |
| 15      | 1048–1141   | Two-Lane Highways                                     |
| 16      | 1142–1194   | Urban Street Facilities                               |
| 17      | 1195–1263   | Urban Street Reliability and ATDM                     |
| 18      | 1264–1396   | Urban Street Segments                                 |
| 19      | 1397–1582   | Signalized Intersections                              |
| 20      | 1583–1682   | Two-Way Stop-Controlled Intersections                 |
| 21      | 1683–1731   | All-Way Stop-Controlled Intersections                 |
| 22      | 1732–1785   | Roundabouts                                           |
| 23      | 1786–1951   | Ramp Terminals and Alternative Intersections          |
| 24      | 1952–2011   | Off-Street Pedestrian and Bicycle Facilities          |
| 25      | 2012–2255   | Freeway Facilities: Supplemental                      |
| 26      | 2256–2457   | Freeway and Highway Segments: Supplemental            |
| 27      | 2458–2530   | Freeway Weaving: Supplemental                         |
| 28      | 2531–2579   | Freeway Merges and Diverges: Supplemental             |
| 29      | 2580–2749   | Urban Street Facilities: Supplemental                 |
| 30      | 2750–2894   | Urban Street Segments: Supplemental                   |
| 31      | 2895–3165   | Signalized Intersections: Supplemental                |
| 32      | 3166–3284   | Stop-Controlled Intersections: Supplemental           |
| 33      | 3285–3325   | Roundabouts: Supplemental                             |
| 34      | 3326–3513   | Interchange Ramp Terminals: Supplemental              |
| 35      | 3514–3526   | Pedestrian and Bicycles: Supplemental                 |
| 36      | 3527-3609   | Concepts: Supplemental                                |
| 37      | 3610–3653   | ATDM: Supplemental                                    |
| 38      | 3654–3957   | Network Analysis                                      |

## Note
Note that the splitted PDF must be readable format. Sometime, they are converted to images, which are not readable by the text extraction tools. In that case, you can use OCR tools like Tesseract to convert them to text (not implemented yet).