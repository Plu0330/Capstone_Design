curl -G "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode" \
    --data-urlencode "query=$1" \
    --data-urlencode "coordinate=$2, $3" \
    -H "X-NCP-APIGW-API-KEY-ID: b1hmt28uqq" \
    -H "X-NCP-APIGW-API-KEY: 6L3w8iuRqWW1jznYj7ntZO3y3n5pX1cUnxuuCD7h" -v
