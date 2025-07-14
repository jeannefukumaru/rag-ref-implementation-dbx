from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import displayHTML
from databricks.sdk.service.serving import DataframeSplitInput
import json
from typing import List, Dict


def query_endpoint(query: str, messages: List[Dict], endpoint_name: str) -> List[Dict]:
    w = WorkspaceClient()
    messages.append({"role": "user", "content": query})
    df_split = DataframeSplitInput(
        columns=["messages"],
        data=[[{"messages": messages}]],
    )
    response = w.serving_endpoints.query(endpoint_name, dataframe_split=df_split)
    text_response = response.predictions[0]["result"]
    display_chat(messages, response.predictions[0])
    messages.append({"role": "assistant", "content": text_response})
    return messages


def display_chat(chat_history, response):
    def user_message_html(message):
        return f"""
      <div style="width: 90%; border-radius: 10px; background-color: #c2efff; padding: 10px; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; font-size: 14px;">
        {message}
      </div>"""

    def assistant_message_html(message):
        return f"""
      <div style="width: 90%; border-radius: 10px; background-color: #e3f6fc; padding: 10px; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; margin-left: 40px; font-size: 14px">
        <img style="float: left; width:40px; margin: -10px 5px 0px -10px" src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/robot.png?raw=true"/>
        {message}
      </div>"""

    chat_history_html = "".join(
        [
            user_message_html(m["content"])
            if m["role"] == "user"
            else assistant_message_html(m["content"])
            for m in chat_history
        ]
    )
    answer = response["result"].replace("\n", "<br/>")
    sources_html = (
        (
            "<br/><br/><br/><strong>Sources:</strong><br/> <ul>"
            + "\n".join(
                [
                    f"""<li>{s['doc_name']} - Page {s['page_nr']}</li>"""
                    for s in json.loads(response["sources"])
                ]
            )
            + "</ul>"
        )
        if response["sources"]
        else ""
    )
    response_html = f"""{answer}{sources_html}"""

    displayHTML(chat_history_html + assistant_message_html(response_html))
