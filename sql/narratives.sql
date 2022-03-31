SELECT DISTINCT
	Basic_Incident_Number									AS 'incident_number'
,	Basic_Incident_Type_Code								AS 'incident_type_code'
,	Basic_Incident_Type_Code_And_Description				AS 'incident_type'
,	CONVERT(NVARCHAR(MAX),Basic_Incident_Primary_Narrative)	AS 'narrative'

FROM [DCP-FIRESQL1PRD].[CF_DataMart].[DwFire].[Dim_Basic]

WHERE 
	Basic_Incident_Primary_Narrative IS NOT NULL
AND	Basic_Incident_Validity_Score_Range = '80 - 100'
AND	Basic_Incident_Status IN ('Finalized','Ready for QA')