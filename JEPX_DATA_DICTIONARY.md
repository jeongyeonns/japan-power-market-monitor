# JEPX 데이터 사전

이 문서는 `data/jepx/raw/spot_summary_2026.csv`를 원본 수정 없이 Python으로
검사한 결과입니다. 확인되지 않은 내용은 임의로 추정하지 않습니다.

## 원본 파일

| 항목 | 확인 결과 |
|---|---|
| 파일명 | `spot_summary_2026.csv` |
| 파일 형식 | CSV |
| 파일 크기 | 708,380 bytes |
| SHA-256 | `4B075DB56831E28147A048CB29FBAF400B5AE5F516469925ADF3D3BC07C3A34F` |
| 선택 인코딩 | `cp932` |
| 인코딩 시험 | cp932 성공, shift_jis 성공, utf-8-sig 실패, utf-8 실패 |
| 구분자 | 쉼표(`,`) |
| 헤더 | 첫 번째 행 |
| 열 수 | 19열 |
| 원본 데이터 행 | 5,424행 |
| 데이터 기간 | 2026-04-01 ~ 2026-07-22 |
| 개별 날짜 수 | 113일 |
| 날짜별 시간대 | 모두 48개 |
| 깨진 문자·대체문자 | 확인되지 않음 |

cp932와 shift_jis 모두 현재 파일을 대체문자 없이 디코딩하지만, 정해진 시험 순서에서
먼저 성공한 cp932를 선택했습니다.

## 실제 열 이름과 내부 매핑

| 실제 원본 열 | 내부 wide 열 | 의미·단위 |
|---|---|---|
| 受渡日 | `delivery_date` | 수전일, `YYYY/MM/DD` |
| 時刻コード | `period_no` | 30분 코마 번호, 1~48 |
| 売り入札量(kWh) | `sell_bid_volume_kwh` | 매도 입찰량, kWh |
| 買い入札量(kWh) | `buy_bid_volume_kwh` | 매수 입찰량, kWh |
| 約定総量(kWh) | `contracted_volume_kwh` | 약정 총량, kWh |
| システムプライス(円/kWh) | `system_price` | 시스템가격, 엔/kWh |
| エリアプライス北海道(円/kWh) | `hokkaido_price` | 홋카이도 지역가격, 엔/kWh |
| エリアプライス東北(円/kWh) | `tohoku_price` | 도호쿠 지역가격, 엔/kWh |
| エリアプライス東京(円/kWh) | `tokyo_price` | 도쿄 지역가격, 엔/kWh |
| エリアプライス中部(円/kWh) | `chubu_price` | 중부 지역가격, 엔/kWh |
| エリアプライス北陸(円/kWh) | `hokuriku_price` | 호쿠리쿠 지역가격, 엔/kWh |
| エリアプライス関西(円/kWh) | `kansai_price` | 간사이 지역가격, 엔/kWh |
| エリアプライス中国(円/kWh) | `chugoku_price` | 주고쿠 지역가격, 엔/kWh |
| エリアプライス四国(円/kWh) | `shikoku_price` | 시코쿠 지역가격, 엔/kWh |
| エリアプライス九州(円/kWh) | `kyushu_price` | 규슈 지역가격, 엔/kWh |
| 売りブロック入札総量(kWh) | `sell_block_bid_volume_kwh` | 매도 블록 입찰 총량, kWh |
| 売りブロック約定総量(kWh) | `sell_block_contracted_volume_kwh` | 매도 블록 약정 총량, kWh |
| 買いブロック入札総量(kWh) | `buy_block_bid_volume_kwh` | 매수 블록 입찰 총량, kWh |
| 買いブロック約定総量(kWh) | `buy_block_contracted_volume_kwh` | 매수 블록 약정 총량, kWh |

로더가 추가하는 wide 메타데이터 열은 `period_start`, `datetime_jst`,
`price_unit`, `source_file`, `source_row`, `source_status`입니다.

## 분석용 long 구조

가격 분석에는 시스템가격과 9개 지역가격을 같은 구조로 비교할 수 있는 long 형식을
사용합니다. 원본 5,424행 × 가격 구분 10개 = 54,240행입니다.

| 내부 열 | 의미 |
|---|---|
| `delivery_date` | 표준 날짜 |
| `period_no` | 1~48 코마 |
| `period_start` | `HH:MM`, 00:00~23:30 |
| `datetime_jst` | Asia/Tokyo 시간대가 포함된 일시 |
| `area` | `System`, `Hokkaido`, `Tohoku`, `Tokyo`, `Chubu`, `Hokuriku`, `Kansai`, `Chugoku`, `Shikoku`, `Kyushu` |
| `area_display` | 한국어 표시명 |
| `price` | 숫자로 변환된 가격 |
| `price_unit` | `円/kWh` |
| `original_price_column` | 실제 일본어 원본 가격 열 |
| `standard_price_column` | wide 형식의 영어 가격 열 |
| `raw_price` | 변환 전 원본 가격 값 |
| `source_file` | 원본 파일명 |
| `source_row` | 헤더를 포함한 원본 행 번호 |
| `source_status` | `Valid` 또는 `Error` |

## 날짜와 시간대

- 날짜 원본 형식: `YYYY/MM/DD`
- 시각 원본 형식: `時刻コード` 1~48
- `period_no=1`은 00:00, `period_no=48`은 23:30으로 표준화합니다.
- 각 코마는 실제 파일 구조상 30분 간격입니다.
- `datetime_jst`의 시간대는 Asia/Tokyo입니다.

## 가격 및 지역

- 시스템가격 열이 별도로 존재합니다.
- 9개 지역가격 열이 wide 형식으로 존재합니다.
- 가격 단위는 모든 가격 열 이름에 명시된 `円/kWh`입니다.
- 시스템가격과 지역가격의 산술적 관계나 가격 결정 규칙은 이 파일만으로 확인하지
  않았으며, 추가 공식 문서 확인이 필요합니다.

## 품질 검사 결과

- 날짜·시간대 중복: 0건
- 가격 결측: 0건
- 숫자 변환 실패: 0건
- 음수 가격: 0건
- 동일 키의 서로 다른 가격: 0건
- 날짜별 48개 코마 누락: 0건
- 가격 최솟값·최댓값: 0.01~64.28 엔/kWh
- 원본에서 사용한 별도 결측 표기: 확인되지 않음
- 공식 가격 상·하한과 이상치 판정 기준: 확인 필요
- 데이터 상태(속보치·확정치·수정본 여부): 파일 열에서 확인되지 않아 확인 필요

## 원본 행 확인

첫 10행과 마지막 5행을 19개 열 전체로 검사했습니다. 첫 행은
`2026/04/01, 時刻コード 1`, 마지막 행은 `2026/07/22, 時刻コード 48`입니다.
첫 행의 시스템가격은 17.75 엔/kWh이고 마지막 행은 17.72 엔/kWh입니다.
검사 과정에서 원본 파일을 수정하거나 다시 저장하지 않았습니다.
