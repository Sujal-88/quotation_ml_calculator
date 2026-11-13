# Data dictionary and input contract

## Required database

- **SQLite table:** `orders`
- **`total_amount`**: numeric historical order amount (regression target; actual currency unverified).
- **`details`**: JSON text or double-encoded JSON text describing printing specifications.
- **`date_created`**: parseable timestamp (needed only for revenue forecasting).

Other tables inspected in the original notebook (`invoices`, `quotations`, `expenses`, `job_cards`) are **not consumed** by the current trained models.

## Enhanced quotation features

| Feature | Data source | Interpretation |
|---|---|---|
| `total_quantity` | `printingSpecs.printingDetails.*.qty`, fallback `paperQuantity`, then `totalQuantity` | Heuristic printing run quantity; verify against actual schema |
| `gsm` | `paperEntries[0].gSM` | Paper weight |
| `binding` | `bindingDetails` | Binding category |
| `has_lamination`, `num_lamination_types` | `lamination` | Lamination indicators and count |
| `has_gloss`, `has_matte` | `lamination` | Lamination type flags |
| `num_paper_entries` | `paperEntries` | Number of paper entries |
| `paper_type` | `paperEntries[0].paperType` | Paper category |
| `paper_supplied_by` | `paperEntries[0].paperSuppliedBy` | Supply source |
| `paper_purchase_from` | `paperEntries[0].paperPurchaseFrom` | Purchase source |
| `sheet_size` | `paperEntries[0].sheetSize` | Raw sheet size string |
| `ctp_supplied_by` | `paperEntries[0].printingSpecs.ctpSuppliedBy` | CTP supplier/source |
| `ctp_sets` | `paperEntries[0].printingSpecs.totalSetsofCtp` | Number of CTP sets |
| `printing_type` | first entry of `printingSpecs.printingType` | Printing mode |

## Simplified model features

| Feature | Meaning |
|---|---|
| `sheets_qty` | `paperEntries[0].paperQuantity` |
| `job_size` | `paperEntries[0].sheetSize` |
| `print_side` | `FB` if printing mode indicates back/both/FB/double; otherwise `Single Sided` |
| `job_type` | Inferred `Multi Page` when binding is not `No Binding`; otherwise `Single Page` |

**Important:** These features are reconstructed from the uploaded prototype's assumptions, not a full documented production ERP schema. Unexpected JSON input or invalid amounts are excluded and counted in the training summary.
