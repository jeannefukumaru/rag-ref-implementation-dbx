import io
import pyspark.sql.functions as func
from pyspark.sql.types import MapType, StringType
from pypdf import PdfReader
import time


def bronze_to_silver_pypdf(df_bronze):
    """
    Parse pdf documents simply through pypdf
    """
    @func.udf(returnType=MapType(StringType(), StringType()))
    def parse_pdf(pdf_content):
        pdf = io.BytesIO(pdf_content)
        reader = PdfReader(pdf)
        return {str(count): page.extract_text() for count, page in enumerate(reader.pages)}

    df_parsed = df_bronze.select(
        func.col("input_file"),
        parse_pdf("content").alias("parsed_pdf_pages"),
    )

    return (
        df_parsed.select("*", func.explode("parsed_pdf_pages"))
        .withColumnRenamed("key", "page_nr")
        .withColumnRenamed("value", "page_content")
        .drop("parsed_pdf_pages")
    )

def index_exists(vsc, endpoint_name, index_full_name):
    try:
        dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
        return dict_vsindex.get("status").get("ready", False)
    except Exception as e:
        if "RESOURCE_DOES_NOT_EXIST" not in str(e):
            print(
                f"Unexpected error describing the index. This could be a permission issue."
            )
            raise e
    return False


def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
    for i in range(180):
        idx = vsc.get_index(vs_endpoint_name, index_name).describe()
        index_status = idx.get("status", idx.get("index_status", {}))
        status = index_status.get(
            "detailed_state", index_status.get("status", "UNKNOWN")
        ).upper()
        url = index_status.get("index_url", index_status.get("url", "UNKNOWN"))
        if "ONLINE" in status:
            return
        if "UNKNOWN" in status:
            print(
                f"Can't get the status - will assume index is ready {idx} - url: {url}"
            )
            return
        elif "PROVISIONING" in status:
            if i % 40 == 0:
                print(
                    f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}"
                )
            time.sleep(10)
        else:
            raise Exception(
                f"""Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}"""
            )
    raise Exception(
        f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}"
    )

