drop table if exists myCsv, myDupes;

create temporary table myCsv (
    "timestamp" text
    ,bid    float
    ,ask    float
    ,vol    int
);

copy myCsv
from {{path}} delimiter ',' csv header;

select 0::int AS "dupe_count", * into myDupes from myCsv limit(0);
select * into myOkays from myCsv limit(0);

with dupes as (
    select "timestamp", count(*) as "dupe_count"
    from myCsv
    group by "timestamp"
    having count(*) > 1
)
insert into myDupes
select d."dupe_count", mc.* 
from myCsv as mc
join dupes d on d."timestamp" = mc."timestamp";

insert into myOkays
select mc.* 
from myCsv as mc
where not exists ( 
    select 1
    from myDupes d 
    where d."timestamp" = mc."timestamp"
);