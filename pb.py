# -*- coding: utf-8 -*-

aa = Pb()
aa.key
getattr(aa,key)
aa.HasField(key)
from google.protobuf.json_format import MessageToJson
MessageToJson(aa)
空
isinstance(aa, _message.Message)
from google.protobuf import message as _message

f[0].name for f in pb.ListFields()
isinstance(f[0], FieldDescriptor)

import json
json.loads(MessageToJson(aa, including_default_value_fields = True))
aa.DESCRIPTOR.fields
field.containing_oneof
field.label
field.name
field.cpp_type

descriptor.FieldDescriptor.CPPTYPE_MESSAGE
descriptor.FieldDescriptor.LABEL_REPEATED = 3    REPEATED字段

field.label = 1 optional
2 required
field.cpp_type = 1 int型
6 float型
9 string型
10 pb型

aa.IsInitialized()
required字段赋值，即为True

repeated Pb info：继续往下？

