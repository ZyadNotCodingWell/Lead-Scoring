# Salesforce Integration Guide

## Architecture Overview

```
Salesforce Org
│
├── Lead Insert / Update
│       └── LeadScoreTrigger.trigger
│               └──(async)── LeadScoringAPI.scoreLeadsAsync()
│                               └── POST /ingest/score/batch ──> ML model
│                                       └── writes Lead_Score__c, Lead_Bucket__c
│
├── Incoming EmailMessage on Lead
│       └── EmailIntentTrigger.trigger
│               └──(async)── LeadScoringAPI.analyzeEmailAsync()
│                               └── POST /email/analyze ──> MNLI + blend
│                                       └── writes Lead_Score__c (blended)
│
└── Scheduled Apex (every 8 h)
        └── AssignmentExportScheduler.execute()
                └──(async)── LeadScoringAPI.triggerAssignmentExportAsync()
                                └── POST /export/trigger
                                        └── saves output/assignments_YYYYMMDD_HHMMSS.csv
```

---

## Part 1 — Salesforce Setup

### Step 1: Add Custom Fields to the Lead Object

Go to **Setup → Object Manager → Lead → Fields & Relationships → New** and create each field below.

| Label              | API Name                 | Type            | Length / Precision | Notes                              |
|--------------------|--------------------------|-----------------|--------------------|------------------------------------|
| Lead Score         | `Lead_Score__c`          | Number          | 18, 4              | 0.0000 – 1.0000                    |
| Lead Bucket        | `Lead_Bucket__c`         | Text            | 10                 | hot / warm / low / cold            |
| Last Scored        | `Last_Scored__c`         | Date/Time       | —                  | Timestamp of last API call         |
| Assigned Rep       | `Assigned_Rep__c`        | Text            | 100                | Populated by export job (optional) |
| Email Opens        | `Email_Opens__c`         | Number          | 18, 0              | From Pardot / Marketing Cloud      |
| Website Visits     | `Website_Visits__c`      | Number          | 18, 0              | From Pardot / Marketing Cloud      |
| Form Submissions   | `Form_Submissions__c`    | Number          | 18, 0              | Web-to-Lead submissions            |
| Engagement Score   | `Engagement_Score__c`    | Number          | 18, 2              | 0–100 composite engagement         |

After creating the fields, add them to the **Lead page layout** so reps can see them.

---

### Step 2: Remote Site Settings

Salesforce must explicitly whitelist any external URL before making HTTP callouts.

1. **Setup → Security → Remote Site Settings → New Remote Site**
2. Fill in:
   - Remote Site Name: `LeadScoringAPI`
   - Remote Site URL: `https://your-api-server.com` (or `http://` for sandboxes only)
   - Active: ✓ checked
3. Save.

> **Production note:** Salesforce enforces HTTPS for all callouts in production orgs.
> Deploy your API behind an HTTPS reverse proxy (nginx, Caddy, or an AWS/GCP load balancer)
> before going to production.

---

### Step 3: Named Credentials

Named Credentials store the base URL and credentials so Apex code never hardcodes them.

1. **Setup → Security → Named Credentials → New**
2. Fill in:
   - Label: `Lead Scoring API`
   - Name: `LeadScoringAPI`  ← this becomes `callout:LeadScoringAPI` in Apex
   - URL: `https://your-api-server.com`
   - Identity Type: `Anonymous`
   - Authentication Protocol: `No Authentication`
3. Save.

> **For production:** Switch to a custom header (`X-API-Key`) or OAuth 2.0 Client
> Credentials flow. Add the header in Named Credentials under "Custom Headers."

---

## Part 2 — Historical Data Export & Model Training

The classifier is trained on real Salesforce leads with `IsConverted` labels. Do this once before go-live, then repeat whenever you want to retrain.

### Step 4: Export Historical Leads from Salesforce

Run the following SOQL in **Data Loader (Export)** or **Workbench → SOQL Query**:

```sql
SELECT
    Id,
    NumberOfEmployees,
    AnnualRevenue,
    Title,
    Industry,
    LeadSource,
    CreatedDate,
    IsConverted,
    Email_Opens__c,
    Website_Visits__c,
    Form_Submissions__c,
    Engagement_Score__c
FROM Lead
WHERE CreatedDate >= LAST_N_YEARS:2
ORDER BY CreatedDate ASC
```

Export as CSV. Aim for at least **500–1000 converted leads** for a meaningful classifier.

---

### Step 5: Ingest the Export into the Pipeline

Send the exported leads to the API (one batch per CSV chunk). Using curl:

```bash
curl -X POST https://your-api-server.com/ingest/leads \
     -H "Content-Type: application/json" \
     -d '{
       "mode": "replace",
       "leads": [
         {
           "salesforce_id": "00Q000000000001AAA",
           "number_of_employees": 350,
           "annual_revenue": 45000000,
           "title": "VP of Engineering",
           "industry": "Technology",
           "lead_source": "Web",
           "created_date": "2023-06-15T10:30:00Z",
           "is_converted": true,
           "email_opens": 4,
           "website_visits": 12,
           "form_submissions": 1,
           "engagement_score": 72.5
         }
       ]
     }'
```

Or automate from a Python script that reads the exported CSV:

```python
import pandas as pd, requests, json

df = pd.read_csv("sf_export.csv")

leads = []
for _, row in df.iterrows():
    leads.append({
        "salesforce_id":      str(row.get("Id", "")),
        "number_of_employees":int(row["NumberOfEmployees"]) if pd.notna(row.get("NumberOfEmployees")) else None,
        "annual_revenue":     float(row["AnnualRevenue"])   if pd.notna(row.get("AnnualRevenue"))     else None,
        "title":              str(row.get("Title", ""))     if pd.notna(row.get("Title")) else None,
        "industry":           str(row.get("Industry", ""))  if pd.notna(row.get("Industry")) else None,
        "lead_source":        str(row.get("LeadSource", ""))if pd.notna(row.get("LeadSource")) else None,
        "created_date":       str(row.get("CreatedDate", "")),
        "is_converted":       bool(row.get("IsConverted", False)),
        "email_opens":        int(row.get("Email_Opens__c", 0) or 0),
        "website_visits":     int(row.get("Website_Visits__c", 0) or 0),
        "form_submissions":   int(row.get("Form_Submissions__c", 0) or 0),
        "engagement_score":   float(row.get("Engagement_Score__c", 50) or 50),
    })

# Send in batches of 200
BATCH = 200
for i in range(0, len(leads), BATCH):
    resp = requests.post(
        "https://your-api-server.com/ingest/leads",
        json={"mode": "append" if i > 0 else "replace", "leads": leads[i:i+BATCH]},
    )
    print(resp.json())
```

---

### Step 6: Train the Classifier

```bash
# On the API server
python models/train.py
```

This reads `data/leads.csv`, trains a `GradientBoostingClassifier`, evaluates it,
and saves the model to `models/lead_scorer.pkl`. Re-run this any time you push a
fresh batch of Salesforce leads.

---

## Part 3 — Apex Code

Create these four files in your Salesforce org via **Developer Console** or an IDE (VS Code + Salesforce Extension Pack).

---

### File 1: `LeadScoringAPI.cls`

This is the single class that handles all callouts. All methods are `@future` so
they run asynchronously and don't block Lead DML operations.

```apex
public class LeadScoringAPI {

    static final String BASE = 'callout:LeadScoringAPI';

    // ── Real-time lead scoring ────────────────────────────────────────────────
    // Called from LeadScoreTrigger. Scores a batch of leads in one HTTP callout.

    @future(callout=true)
    public static void scoreLeadsAsync(List<Id> leadIds) {
        List<Lead> leads = [
            SELECT Id, NumberOfEmployees, AnnualRevenue, Title, Industry,
                   LeadSource, CreatedDate,
                   Email_Opens__c, Website_Visits__c,
                   Form_Submissions__c, Engagement_Score__c
            FROM   Lead
            WHERE  Id IN :leadIds
            AND    IsConverted = false
        ];
        if (leads.isEmpty()) return;

        List<Object> payloadList = new List<Object>();
        for (Lead l : leads) {
            payloadList.add(buildPayload(l));
        }

        HttpRequest req = new HttpRequest();
        req.setEndpoint(BASE + '/ingest/score/batch');
        req.setMethod('POST');
        req.setHeader('Content-Type', 'application/json');
        req.setBody(JSON.serialize(new Map<String, Object>{ 'leads' => payloadList }));
        req.setTimeout(30000);

        HttpResponse res = new Http().send(req);
        if (res.getStatusCode() != 200) {
            System.debug('LeadScoringAPI.scoreLeadsAsync error: ' + res.getStatus());
            return;
        }

        Map<String, Object> body = (Map<String, Object>) JSON.deserializeUntyped(res.getBody());
        List<Object> results     = (List<Object>) body.get('results');

        Map<String, Object> scoreById = new Map<String, Object>();
        for (Object r : results) {
            Map<String, Object> sr = (Map<String, Object>) r;
            scoreById.put((String) sr.get('salesforce_id'), sr);
        }

        List<Lead> toUpdate = new List<Lead>();
        for (Lead l : leads) {
            if (scoreById.containsKey(l.Id)) {
                Map<String, Object> sr = (Map<String, Object>) scoreById.get(l.Id);
                l.Lead_Score__c  = (Decimal) sr.get('score');
                l.Lead_Bucket__c = (String)  sr.get('bucket');
                l.Last_Scored__c = Datetime.now();
                toUpdate.add(l);
            }
        }
        if (!toUpdate.isEmpty()) update toUpdate;
    }


    // ── Email intent analysis ────────────────────────────────────────────────
    // Called from EmailIntentTrigger. Analyses the reply and blends the score.
    // leadIdsJson / emailBodiesJson: JSON-serialised List<String> (parallel arrays).

    @future(callout=true)
    public static void analyzeEmailAsync(String leadIdsJson, String emailBodiesJson) {
        List<String> leadIds     = (List<String>) JSON.deserialize(leadIdsJson,     List<String>.class);
        List<String> emailBodies = (List<String>) JSON.deserialize(emailBodiesJson, List<String>.class);

        List<Lead> toUpdate = new List<Lead>();

        for (Integer i = 0; i < leadIds.size(); i++) {
            Map<String, Object> payload = new Map<String, Object>{
                'email_body' => emailBodies[i],
                'lead_id'    => leadIds[i]
            };

            HttpRequest req = new HttpRequest();
            req.setEndpoint(BASE + '/email/analyze');
            req.setMethod('POST');
            req.setHeader('Content-Type', 'application/json');
            req.setBody(JSON.serialize(payload));
            req.setTimeout(120000);  // MNLI can be slow on first call

            HttpResponse res = new Http().send(req);
            if (res.getStatusCode() != 200) {
                System.debug('analyzeEmailAsync error for ' + leadIds[i] + ': ' + res.getStatus());
                continue;
            }

            Map<String, Object> result = (Map<String, Object>) JSON.deserializeUntyped(res.getBody());
            Object blendedObj = result.get('blended_score');
            if (blendedObj == null) continue;  // no model score was available

            toUpdate.add(new Lead(
                Id             = leadIds[i],
                Lead_Score__c  = (Decimal) blendedObj,
                Lead_Bucket__c = (String)  result.get('final_bucket'),
                Last_Scored__c = Datetime.now()
            ));
        }

        if (!toUpdate.isEmpty()) update toUpdate;
    }


    // ── Scheduled assignment export ──────────────────────────────────────────
    // Called from AssignmentExportScheduler. Triggers the server-side optimizer.

    @future(callout=true)
    public static void triggerAssignmentExportAsync() {
        HttpRequest req = new HttpRequest();
        req.setEndpoint(BASE + '/export/trigger');
        req.setMethod('POST');
        req.setHeader('Content-Type', 'application/json');
        req.setBody('{}');
        req.setTimeout(120000);

        HttpResponse res = new Http().send(req);
        System.debug('Export trigger response: ' + res.getStatusCode() + ' ' + res.getBody());
    }


    // ── Helper: Salesforce Lead → API payload ────────────────────────────────

    private static Map<String, Object> buildPayload(Lead l) {
        return new Map<String, Object>{
            'salesforce_id'       => l.Id,
            'number_of_employees' => l.NumberOfEmployees,
            'annual_revenue'      => l.AnnualRevenue,
            'title'               => l.Title,
            'industry'            => l.Industry,
            'lead_source'         => l.LeadSource,
            'created_date'        => l.CreatedDate != null
                                         ? String.valueOf(l.CreatedDate)
                                         : null,
            'is_converted'        => false,
            'email_opens'         => l.Email_Opens__c != null ? l.Email_Opens__c.intValue() : 0,
            'website_visits'      => l.Website_Visits__c != null ? l.Website_Visits__c.intValue() : 0,
            'form_submissions'    => l.Form_Submissions__c != null ? l.Form_Submissions__c.intValue() : 0,
            'engagement_score'    => l.Engagement_Score__c != null ? l.Engagement_Score__c : 50.0
        };
    }
}
```

---

### File 2: `LeadScoreTrigger.trigger`

Fires after a Lead is created or updated. Passes IDs to the async scoring method.

```apex
trigger LeadScoreTrigger on Lead (after insert, after update) {
    List<Id> toScore = new List<Id>();

    for (Lead l : Trigger.new) {
        // Only score non-converted leads
        if (!l.IsConverted) {
            // On update: only re-score if a scoring-relevant field changed
            if (Trigger.isInsert) {
                toScore.add(l.Id);
            } else {
                Lead old = Trigger.oldMap.get(l.Id);
                if (l.Title              != old.Title
                 || l.Industry           != old.Industry
                 || l.NumberOfEmployees  != old.NumberOfEmployees
                 || l.AnnualRevenue      != old.AnnualRevenue
                 || l.LeadSource         != old.LeadSource
                 || l.Email_Opens__c     != old.Email_Opens__c
                 || l.Website_Visits__c  != old.Website_Visits__c
                 || l.Form_Submissions__c != old.Form_Submissions__c) {
                    toScore.add(l.Id);
                }
            }
        }
    }

    if (!toScore.isEmpty()) {
        LeadScoringAPI.scoreLeadsAsync(toScore);
    }
}
```

> The update check prevents infinite loops: the `@future` method writes
> `Lead_Score__c` back, which would re-fire the trigger. Since those fields
> are not in the re-score condition, the loop stops after one round.

---

### File 3: `EmailIntentTrigger.trigger`

Fires when a new `EmailMessage` is inserted. Routes incoming lead emails to MNLI.

```apex
trigger EmailIntentTrigger on EmailMessage (after insert) {
    List<String> leadIds     = new List<String>();
    List<String> emailBodies = new List<String>();

    for (EmailMessage em : Trigger.new) {
        // Only process inbound messages related to a Lead record
        if (em.Incoming != true || em.RelatedToId == null) continue;

        Schema.SObjectType objType = em.RelatedToId.getSObjectType();
        if (objType == null) continue;
        if (objType.getDescribe().getName() != 'Lead') continue;

        String body = em.TextBody != null ? em.TextBody : '';
        if (String.isBlank(body)) continue;

        leadIds.add(em.RelatedToId);
        emailBodies.add(body);
    }

    if (!leadIds.isEmpty()) {
        LeadScoringAPI.analyzeEmailAsync(
            JSON.serialize(leadIds),
            JSON.serialize(emailBodies)
        );
    }
}
```

---

### File 4: `AssignmentExportScheduler.cls`

Implements `Schedulable` so Salesforce can run it on a cron schedule.

```apex
global class AssignmentExportScheduler implements Schedulable {
    global void execute(SchedulableContext ctx) {
        LeadScoringAPI.triggerAssignmentExportAsync();
    }
}
```

---

## Part 4 — Scheduling the Assignment Export

### Step 7: Schedule in Salesforce (every 8 hours)

Open **Developer Console → Debug → Open Execute Anonymous Window** and run:

```apex
// Remove any existing job with the same name first
for (CronTrigger ct : [SELECT Id FROM CronTrigger WHERE CronJobDetail.Name = 'LeadAssignmentExport']) {
    System.abortJob(ct.Id);
}

// Schedule at 00:00, 08:00, and 16:00 every day
// Salesforce cron does not support "every N hours" natively, so schedule 3 jobs.
AssignmentExportScheduler job = new AssignmentExportScheduler();
System.schedule('LeadAssignmentExport_00', '0 0 0  * * ?', job);
System.schedule('LeadAssignmentExport_08', '0 0 8  * * ?', job);
System.schedule('LeadAssignmentExport_16', '0 0 16 * * ?', job);
```

To verify: **Setup → Scheduled Jobs** — you should see three entries.

---

### Step 8: Server-side cron (alternative / fallback)

If you prefer not to use Salesforce Scheduled Apex (e.g., no callout permissions in your
edition), run the standalone script directly on the API server.

**Linux/macOS — crontab:**
```cron
# Edit with: crontab -e
0 0,8,16 * * * cd /path/to/Lead_Scoring-main && python jobs/scheduled_assignment.py >> /var/log/lead_assignment.log 2>&1
```

**Windows — Task Scheduler (PowerShell):**
```powershell
$action  = New-ScheduledTaskAction -Execute "python" `
           -Argument "jobs\scheduled_assignment.py" `
           -WorkingDirectory "C:\path\to\Lead_Scoring-main"
$trigger = New-ScheduledTaskTrigger -Daily -At "00:00" -RepetitionInterval (New-TimeSpan -Hours 8)
Register-ScheduledTask -TaskName "LeadAssignmentExport" -Action $action -Trigger $trigger -RunLevel Highest
```

Both approaches write a timestamped CSV to `output/assignments_YYYYMMDD_HHMMSS.csv`.

---

### Step 9: Configure the Team Roster

Edit `config/optimizer.json` to match your actual sales team before the first run:

```json
{
  "salespeople": [
    {"name": "Alice Smith",    "role": "salesman", "capacity": 20},
    {"name": "Bob Jones",      "role": "salesman", "capacity": 20},
    {"name": "Carol (Senior)", "role": "senior",   "capacity": 10}
  ],
  "industry_quotas": {
    "Technology":    5,
    "Finance":       3,
    "Healthcare":    2,
    "Manufacturing": 2,
    "Retail":        2
  }
}
```

**Role routing:**
- `"salesman"` → assigned leads with seniority **Individual Contributor** or **Manager**
- `"senior"`   → assigned leads with seniority **Director**, **VP**, or **C-Suite**

---

## Part 5 — API Endpoint Reference

| Method | Path | Called From | Description |
|--------|------|-------------|-------------|
| `POST` | `/ingest/score` | Apex (single) | Score one SF lead, return score + bucket |
| `POST` | `/ingest/score/batch` | Apex (batch) | Score a list of SF leads in one callout |
| `POST` | `/ingest/leads` | Python script | Bulk-import training data from SF export |
| `POST` | `/email/analyze` | Apex trigger | MNLI intent + 60/40 score blend |
| `POST` | `/score/lead` | Ad hoc | Score using internal feature schema |
| `POST` | `/score/optimize` | Ad hoc | Run optimizer via API |
| `POST` | `/export/trigger` | Apex scheduler | Run optimizer, save CSV to output/ |
| `GET`  | `/export/assignments` | Browser / curl | Download latest assignment CSV |
| `GET`  | `/health` | Monitoring | Liveness check |

Interactive docs (Swagger UI): `https://your-api-server.com/docs`

---

## Part 6 — Testing & Validation

### Test 1: Liveness
```bash
curl https://your-api-server.com/health
# Expected: {"status":"ok","service":"salesforce-lead-scoring","version":"0.2.0"}
```

### Test 2: Single lead scoring
```bash
curl -X POST https://your-api-server.com/ingest/score \
     -H "Content-Type: application/json" \
     -d '{
       "salesforce_id": "00Q000000000001AAA",
       "number_of_employees": 450,
       "annual_revenue": 60000000,
       "title": "VP of Sales",
       "industry": "Technology",
       "lead_source": "Event",
       "created_date": "2024-01-15T09:00:00Z",
       "is_converted": false,
       "email_opens": 6,
       "website_visits": 20,
       "form_submissions": 2,
       "engagement_score": 78.0
     }'
# Expected: {"salesforce_id":"00Q000000000001AAA","score":0.xxxx,"bucket":"warm"}
```

### Test 3: Email intent + score blend
```bash
curl -X POST https://your-api-server.com/email/analyze \
     -H "Content-Type: application/json" \
     -d '{
       "email_body": "Hi, we are definitely interested in scheduling a demo next week. Budget has been approved.",
       "lead_id": "00Q000000000001AAA"
     }'
# Expected: blended_score, final_bucket, score_updated: true
```

### Test 4: Manual assignment export
```bash
curl https://your-api-server.com/export/assignments --output assignments.csv
```

### Test 5: Trigger the scheduled export
```bash
curl -X POST https://your-api-server.com/export/trigger
# Expected: {"status":"optimal","total_leads_assigned":...,"saved_to":"output/..."}
```

---

## Production Checklist

- [ ] API deployed behind HTTPS (required by Salesforce production orgs)
- [ ] Named Credential URL updated to HTTPS
- [ ] API key / OAuth header added to Named Credential
- [ ] Historical leads ingested and model retrained (`python models/train.py`)
- [ ] `config/optimizer.json` updated with real team roster
- [ ] Custom Lead fields added and visible on page layout
- [ ] Remote Site Settings pointing to production URL
- [ ] Three Scheduled Apex jobs active (00:00, 08:00, 16:00)
- [ ] Trigger test: create a Lead in Salesforce, verify `Lead_Score__c` populates within ~30 s
- [ ] Email test: send a reply to a Lead email thread, verify score updates
- [ ] Assignment CSV appearing in `output/` every 8 hours
