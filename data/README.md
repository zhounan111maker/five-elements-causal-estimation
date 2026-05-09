# 数据集说明

## BankChurners.csv

- **来源**：Kaggle 银行客户流失数据集（Bank Customer Churn Dataset）
- **样本量**：约 10,000 条记录
- **用途**：客户流失预测与因果效应估计

### 核心变量

| 列名 | 类型 | 说明 |
|------|------|------|
| CLIENTNUM | 标识 | 客户编号 |
| Attrition_Flag | 分类 | 客户状态（Existing / Attrited） |
| Customer_Age | 数值 | 客户年龄 |
| Gender | 分类 | 性别（M / F） |
| Dependent_count | 数值 | 家属数量 |
| Education_Level | 分类 | 教育水平 |
| Marital_Status | 分类 | 婚姻状况 |
| Income_Category | 分类 | 收入分类（<40K / 40K-60K / ... / 120K+ / Unknown） |
| Card_Category | 分类 | 卡类型（Blue / Silver / Gold / Platinum） |
| Months_on_book | 数值 | 在册月数 |
| Total_Relationship_Count | 数值 | 产品关系数 |
| Months_Inactive_12_mon | 数值 | 12月内不活跃月数 |
| Contacts_Count_12_mon | 数值 | 12月内联系次数 |
| Credit_Limit | 数值 | 信用额度 |
| Total_Revolving_Bal | 数值 | 循环余额 |
| Avg_Open_To_Buy | 数值 | 平均可用额度 |
| Total_Amt_Chng_Q4_Q1 | 数值 | Q4/Q1 交易金额变化比 |
| Total_Trans_Amt | 数值 | 总交易金额 |
| Total_Trans_Ct | 数值 | 总交易次数 |
| Total_Ct_Chng_Q4_Q1 | 数值 | Q4/Q1 交易次数变化比 |
| Avg_Utilization_Ratio | 数值 | 平均使用率 |

### 预处理说明

- 数值列无缺失值，可直接使用
- 分类列使用 one-hot 编码处理
- 实验中将 `Total_Trans_Amt` 作为基线结果变量，构造+15%处理效应生成仿真数据
