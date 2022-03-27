import base64
import tensorflow_serving
import requests
import json
from census_consumer_complaint_utils.utils import _bytes_feature, _float_feature, _int64_feature
import tensorflow as tf
from census_consumer_complaint_component.feature_engineering.feature_engineering import TEXT_FEATURES

HOST = "localhost"
PORT = "8501"

URL = f"http://{HOST}:{PORT}/v1/models"
from collections import namedtuple

CensusComplaintRecord = namedtuple("CensusComplaintData", [
    "data_received",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "consumer_complaint_narrative",
    "company_public_response",
    "company",
    "state",
    "zip_code",
    "tag",
    "consumer_consent_provided",
    "submitted_via",
    "date_sent_to_company",
    "company_response_to_consumer",
    "timely_response",
    "complaint_id",

])


def get_serialized_examples(data: CensusComplaintRecord):
    example = tf.train.Example(features=tf.train.Features(
        feature={
            "Date received": _bytes_feature(data.data_received),
            "Product": _bytes_feature(data.product),
            "Sub-product": _bytes_feature(data.sub_product),
            "Issue": _bytes_feature(data.issue),
            "Sub-issue": _bytes_feature(data.sub_issue),
            "Consumer complaint narrative": _bytes_feature(data.consumer_complaint_narrative),
            "Company public response": _bytes_feature(data.company_public_response),
            "Company": _bytes_feature(data.company),
            "State": _bytes_feature(data.state),
            "ZIP code": _float_feature(data.zip_code),

            "Tags": _bytes_feature(data.tag),
            "Consumer consent provided?": _bytes_feature(data.consumer_consent_provided),
            "Submitted via": _bytes_feature(data.submitted_via),

            "Date sent to company": _bytes_feature(data.date_sent_to_company),
            "Company response to consumer": _bytes_feature(data.company_response_to_consumer),
            "Timely response?": _bytes_feature(data.timely_response),
            "Complaint ID": _int64_feature(data.complaint_id),
        }))

    return {"serialized_data": example.SerializeToString()}

def get_rest_request(data=None, model_name="my_model"):
    url = f"{URL}/{model_name}:predict"
    url="http://tf-serving-load-balancer-281291316.ap-south-1.elb.amazonaws.com/v1/models/saved_models:predict"
    headers = {"content-type": "application/json"}
    census_complaint_record = CensusComplaintRecord(
        data_received="2016-05-09",
        product="Mortgage",
        sub_product="Other mortgage",
        issue="Loan servicing, payments, escrow account",
        sub_issue="",
        consumer_complaint_narrative="""We have spoken with several agents of Real Time Resolutions ( RTR ) in regards to the above mentioned mortgage debt being only serviced on behalf of XXXX. ( XXXX ) In response to notices received from RTR dated XXXX XXXX, XXXX, we are requesting that these omissions of RTR having received either Notice of Assignment, Sale or Transfer of servicing rights of the loan to be definitely defined per statutes of Real Estate Settlement Procedures Act ( RESPA ), the Fair Debt collections Act ( FDCPA ), as well as the Fair Credit Reporting Act ( FCRA ) .RTR was served with a Qualified Written Request ( QWR ) on XXXX XXXX, XXXX and they have failed to properly respond to our demands under RESPA , FDCPA and FCRA. They have failed to indicate and establish proper ownership and rights to service the loan. Furthermore, they have failed to provide an itemized accounting ledger of crediting payments made, late fees, etc. from multiple alleged prior servicers of the loan. The assignments, sale affidavits, allonge ( s ) to the note and loan transfers documentation is incomplete and inconsistent with proper mortgage securities and we feel that we are victims of mortgage securitization fraud on many levels, riddled with an array of abusive deceptive practices and misrepresentation.XXXX of RTR, signed an alleged executed agreement with XXXX XXXX XXXX XXXX ( XXXX ) dated XXXX, XXXX, XXXX. By RTR 's own omissions they did not take over servicing rights to the loan until XXXX XXXX, XXXX from XXXX. We have also taken this matter a step further and contacted XXXX County land records here in Texas, and there is no record of assignments for this loan which support or remotely indicate that RTR nor XXXX carry legitimate custodial servicing rights or ownership to this loan. In fact, it appears as though XXXX, still owns the loan which is impossible, because they filed Chapter XXXX bankruptcy in XXXX and all securities liquidated. ( SEE EXHIBIT " A '' -Collection Agreement Forgery ) According to land records, the loan was originated with XXXX . From there, it was transferred to XXXX XXXX XXXX, and XXXX XXXX XXXX was shut down, following XXXX XXXX filing bankruptcy XXXX XXXX, XXXX. So how is it possible that there was any such agreement made with XXXX and RTR? The answer is. it 's not possible.It appears that the agent of RTR, named XXXX XXXX, is in fact a Robo-signer ; since the alleged agreement according to RTR was signed and dated XXXX, XXXX XXXX ; long after the collapse of XXXX. The milestones report that was provided from RTR is dated XXXX XXXX, XXXX. The report documents a span from XXXX XXXX, XXXX until XXXX XXXX, XXXX ; indicating custodial service rights transfers for XXXX, XXXX, and even XXXX XXXX XXXX ( XXXX ) .The problem as previously addressed, is that XXXX dissolved in XXXX, XXXX XXXX Chapter XXXX in XXXX, all securities sold off, and XXXX is prohibited from participating in any consumer loan in the capacity of owner servicer or any such activities as of XXXX. As far as we are able to tell, the milestone report has been falsified. ( SEE EXHIBIT " B '' -- XXXX XXXX ) RTR, has also failed to provide evidence of documentation detailing updated assignments of ownership or necessary proof of their custodial servicing obligations. Furthermore, per RESPA and the FDCPA surrounding mortgage debt validation, we are demanding a complete chain of title as well as the Allonge to the Note, sale affidavits and evidence transfer from XXXX to XXXX and XXXX to XXXX, and finally XXXX to RTR. ( including all accounting ledgers from each servicer previously listed as well. ) In particular, there are a multitude of concerns about the facts surrounding Case # XXXX XXXX which involves CEO XXXX XXXX of XXXX XXXX XXXX ( XXXX ), engaging in a number of unlawful lending practices and abusive collection practices.""",
        company_public_response="",
        company="Real Time Group, Inc.",
        state="TX",
        zip_code=76248,
        tag="",
        consumer_consent_provided="Consent provided",
        submitted_via="Web",
        date_sent_to_company="2016-05-10",
        timely_response="yes",
        company_response_to_consumer="Closed with explanation",
        complaint_id=1916197,
    )
    data = get_serialized_examples(census_complaint_record)
    payload = {
        "signature_name": "serving_default",
        "instances": [
            {
                "examples": {"b64": base64.b64encode(data["serialized_data"]).decode('utf-8')}
            },

        ]
    }
    data = json.dumps(payload)
    response = requests.post(url=url, data=data, headers=headers)
    print(response.content)
    return response


if __name__ == "__main__":
    get_rest_request()
